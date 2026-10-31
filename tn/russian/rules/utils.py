# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2026 Zhendong Peng (pzd17@tsinghua.org.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Russian alphabet and digit-to-word number-name graphs.

NeMo composes binary Gorman/Sproat factorization archives with case lexicons.
These graphs rebuild the same idea natively: a positional factorization maps a
canonical digit string to number-name leaves ("1013" -> "1000 13"), and case
lexicons spell the leaves. Every word of one number shares a grammatical case,
mirroring NeMo's over-generating candidate set, while an extra grammatical
nominative path (agreement-correct "две тысячи", "пять тысяч") is preferred.
"""

from collections import defaultdict
from functools import lru_cache

import pynini
from pynini.lib import byte, pynutil

from tn.utils import get_abs_path, load_labels

RU_LOWER = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
RU_LOWER_ALPHA = pynini.union(*RU_LOWER).optimize()
RU_UPPER_ALPHA = pynini.union(*RU_LOWER.upper()).optimize()
RU_ALPHA = (RU_LOWER_ALPHA | RU_UPPER_ALPHA).optimize()
RU_ALPHA_OR_SPACE = (RU_ALPHA | " ").optimize()

DIGIT = byte.DIGIT
NONZERO = pynini.difference(DIGIT, "0").optimize()
CASES = ("1_cardinals_nominative", "2_cardinals_genitive", "3_cardinals_dative",
         "4_cardinals_accusative", "5_cardinals_instrumental", "6_cardinals_prepositional")
SCALES = ("1000", "1000000", "1000000000")
# Grammatical nominative scale words for a bare scale ("тысяча", "один миллион")
# and after a group ending in 1, 2-4 or 5-9/0/teens.
SCALE_FORMS = {
    "1000": ("тысяча", "тысяча", "тысячи", "тысяч"),
    "1000000": ("один миллион", "миллион", "миллиона", "миллионов"),
    "1000000000": ("один миллиард", "миллиард", "миллиарда", "миллиардов"),
}
FEMININE = {"1": "одна", "2": "две"}


def data_path(name):
    return get_abs_path("russian/data/" + name)


def ordered_rows(name):
    """Rows of a NeMo table with ties broken by listing order.

    NeMo leaves many readings of one key equally weighted and lets audio pick;
    a tiny per-repeat cost makes the ranking (and every n-best list) stable.
    """
    seen = defaultdict(int)
    rows = []
    for row in load_labels(data_path(name)):
        weight = float(row[2]) if len(row) > 2 else 0.0
        output = row[1] if len(row) > 1 else row[0]
        rows.append((row[0], output, str(weight + 0.001 * seen[row[0]])))
        seen[row[0]] += 1
    return rows


def ordered_map(name):
    return pynini.string_map(ordered_rows(name)).optimize()


def latin_to_cyrillic():
    """Latin letter names, e.g. "b" -> "би" (preferred) or "бэ"."""
    return ordered_map("latin_to_cyrillic.tsv")


def _space_join(graph):
    return (graph + pynini.closure(pynini.accep(" ") + graph)).optimize()


class NumberNames:
    """Digit-string to Russian number-name transducers.

    Inputs are canonical digit strings (no separators or leading zeros) of
    up to twelve digits. Leaves carry an agreement tag (``1ж`` for a feminine
    unit before "тысяча", ``1000:2`` for the scale form after 2-4); the tag is
    stripped before the case lexicons of the free, NeMo-like readings.
    """

    def __init__(self):
        self.factorization = self._factorization()
        self.ordinal_factorization = (
            self.factorization | pynutil.add_weight(self._complex_thousands(), -1)
        ).optimize()

        tables = [ordered_rows("numbers/{}.tsv".format(name)) for name in CASES]
        rich_to_plain = {leaf: leaf for leaf in self._plain_leaves()}
        rich_to_plain.update({unit + "ж": unit for unit in FEMININE})
        for scale in SCALES:
            for index in (0, 1, 2, 5):
                rich_to_plain["{}:{}".format(scale, index)] = scale
        strip_leaf = pynini.string_map(sorted(rich_to_plain.items()))
        strip = _space_join(strip_leaf)

        alternatives = {}
        for _, phrase in load_labels(data_path("numbers/cardinals_alternatives.tsv")):
            alternatives.setdefault(phrase.split()[1], phrase)
        alternative_map = pynini.string_map(sorted(alternatives.items()))

        free = []
        for index, rows in enumerate(tables):
            table = pynini.string_map(rows).optimize()
            first = table | (table @ alternative_map)
            if index == 0:
                free.append(pynutil.add_weight(first + pynini.closure(" " + table), -0.1))
            else:
                word = pynutil.add_weight(table, 0.1)
                first = pynutil.add_weight(first, 0.1)
                free.append(first + pynini.closure(" " + word))
        free_cardinal = (strip @ pynini.union(*free)).optimize()

        nominative_rows = tables[0]
        nominative = pynini.string_map(nominative_rows).optimize()
        nominative_first = nominative | (nominative @ alternative_map)
        free_nominative = (strip @ (nominative_first + pynini.closure(" " + nominative))).optimize()

        canonical = dict(load_labels(data_path("numbers/cardinals_nominative_case.tsv")))
        canonical.update({unit + "ж": word for unit, word in FEMININE.items()})
        for scale, forms in SCALE_FORMS.items():
            for index, form in zip((0, 1, 2, 5), forms):
                canonical["{}:{}".format(scale, index)] = form
        canonical_word = pynini.string_map(sorted(canonical.items())).optimize()
        canonical_cardinal = pynutil.add_weight(_space_join(canonical_word), -0.3)
        feminine_word = dict(canonical)
        feminine_word.update(FEMININE)
        feminine_last = pynini.string_map(sorted(feminine_word.items()))
        feminine_cardinal = pynutil.add_weight(pynini.closure(canonical_word + " ") + feminine_last, -0.3)

        # Grammatical nominative readings only (masculine / feminine final unit).
        self.canonical = (self.factorization @ canonical_cardinal).optimize()
        self.feminine = (self.factorization @ feminine_cardinal).optimize()
        self.cardinal = (self.factorization @ (canonical_cardinal | free_cardinal)).optimize()
        self.nominative = (self.factorization @ (canonical_cardinal | free_nominative)).optimize()

        ordinal_rows = ordered_rows("numbers/ordinals.tsv")
        ordinal_word = pynini.string_map(ordinal_rows).optimize()
        by_leaf = defaultdict(list)
        for row in ordinal_rows:
            by_leaf[row[0]].append(row[1])
        masculine = []
        for leaf, words in by_leaf.items():
            preferred = [w for w in words if w.endswith(("ый", "ий"))] or [w for w in words if w.endswith("ой")]
            masculine.append((leaf, preferred[0]))
        masculine_word = pynini.string_map(masculine).optimize()
        # The last leaf may also be a complex "5000" leaf, which has no tag.
        last_leaf = (strip_leaf | pynini.union(*by_leaf.keys())).optimize()
        ordinal_strip = (pynini.closure(strip_leaf + " ") + last_leaf).optimize()
        free_ordinal = ordinal_word | nominative_first + pynini.closure(" " + nominative) + " " + ordinal_word
        free_ordinal = ordinal_strip @ free_ordinal
        # Agreeing number words before the ordinal word; the masculine nominative is the default.
        canonical_prefix = pynini.closure(canonical_word + " ")
        canonical_ordinal = (
            pynutil.add_weight(canonical_prefix + (last_leaf @ ordinal_word), -0.3)
            | pynutil.add_weight(canonical_prefix + (last_leaf @ masculine_word), -0.4)
        )
        self.ordinal = (self.ordinal_factorization @ (canonical_ordinal | free_ordinal)).optimize()

    @staticmethod
    def _plain_leaves():
        leaves = [str(i) for i in range(20)]
        leaves += [str(i * 10) for i in range(2, 10)]
        leaves += [str(i * 100) for i in range(1, 10)]
        return leaves + list(SCALES)

    @staticmethod
    def _units(feminine):
        pairs = [(str(i), str(i) + ("ж" if feminine and str(i) in FEMININE else "")) for i in range(1, 10)]
        return pynini.string_map(pairs).optimize()

    def _two_digits(self, feminine):
        """Exactly two digits with value 1..99."""
        units = self._units(feminine)
        upper = pynini.difference(NONZERO, "1")
        return (
            pynutil.delete("0") + units
            | pynini.accep("1") + DIGIT
            | upper + pynini.accep("0")
            | upper + pynutil.insert("0 ") + units
        ).optimize()

    def _three_digits(self, feminine):
        """Exactly three digits with value 1..999."""
        two = self._two_digits(feminine)
        return (
            NONZERO + pynutil.insert("00") + (pynutil.delete("00") | pynutil.insert(" ") + two)
            | pynutil.delete("0") + two
        ).optimize()

    def _top(self, feminine):
        """One to three digits without a leading zero."""
        return (
            self._units(feminine)
            | (NONZERO + DIGIT) @ self._two_digits(feminine)
            | (NONZERO + DIGIT + DIGIT) @ self._three_digits(feminine)
        ).optimize()

    @staticmethod
    def _agreement(group, scale):
        """Appends the agreeing scale leaf; a bare "1" group becomes the scale."""
        ends_one, ends_few, ends_many = agreement_classes()
        ones = pynini.union("1", "001")
        branches = [
            (pynini.difference(ends_one, ones) @ group) + pynutil.insert(" {}:1".format(scale)),
            (ends_few @ group) + pynutil.insert(" {}:2".format(scale)),
            (ends_many @ group) + pynutil.insert(" {}:5".format(scale)),
            pynini.cross(ones, "{}:0".format(scale)),
        ]
        return pynini.union(*branches).optimize()

    def _scaled_top(self, scale):
        top = NONZERO + pynini.closure(DIGIT, 0, 2)
        return (top @ self._agreement(self._top(scale == "1000"), scale)).optimize()

    def _segment(self, scale):
        group = (DIGIT ** 3) @ self._agreement(self._three_digits(scale == "1000"), scale)
        return (pynutil.insert(" ") + group | pynutil.delete("000")).optimize()

    def _units_segment(self):
        return (pynutil.insert(" ") + self._three_digits(False) | pynutil.delete("000")).optimize()

    def _factorization(self):
        last = self._units_segment()
        return (
            pynini.accep("0")
            | self._top(False)
            | self._scaled_top("1000") + last
            | self._scaled_top("1000000") + self._segment("1000") + last
            | self._scaled_top("1000000000") + self._segment("1000000") + self._segment("1000") + last
        ).optimize()

    def _complex_thousands(self):
        """NeMo's "пятитысячный": a final N x 1000 (N = 2..20) is one ordinal leaf."""
        top = pynini.string_map([(str(n), "{}000".format(n)) for n in range(2, 21)])
        segment = pynutil.insert(" ") + pynini.string_map([("{:03d}".format(n), "{}000".format(n)) for n in range(2, 21)])
        return (
            (top
             | self._scaled_top("1000000") + segment
             | self._scaled_top("1000000000") + self._segment("1000000") + segment)
            + pynutil.delete("000")
        ).optimize()


@lru_cache(maxsize=None)
def number_names():
    return NumberNames()


def written_numbers():
    """Written digit formats ("2 300", "2.300", "2300") -> canonical digits."""
    canonical = (pynini.accep("0") | NONZERO + pynini.closure(DIGIT)).optimize()
    head = NONZERO + pynini.closure(DIGIT, 0, 2)
    dotted = head + (pynutil.delete(".") + DIGIT ** 3).plus
    spaced = head + (pynutil.delete(" ") + DIGIT ** 3).plus
    return (
        pynutil.add_weight(canonical, -0.1)
        | pynutil.add_weight(dotted, 0.1)
        | pynutil.add_weight(spaced, 0.1)
    ).optimize()


def agreement_classes():
    """Digit strings whose noun takes the "one" (1, 21), "few" (2-4, 22) or "many" form."""
    tail = pynini.closure(DIGIT)
    not_one = pynini.difference(DIGIT, "1")
    few = pynini.union("2", "3", "4")
    one = (tail + not_one + "1" | "1").optimize()
    few = (tail + not_one + few | few).optimize()
    many = pynini.difference(DIGIT.plus, one | few).optimize()
    return one, few, many


def insertion_labels(name):
    """NeMo's prepare_labels_for_insertion: label type -> weighted " word" inserts."""
    mapping = defaultdict(list)
    for key, value, weight in ordered_rows(name):
        mapping[key].append(pynutil.add_weight(pynutil.insert(" " + value), float(weight)))
    return {key: pynini.union(*values).optimize() for key, values in mapping.items()}
