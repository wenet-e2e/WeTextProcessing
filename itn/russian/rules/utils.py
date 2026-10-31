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
"""Russian alphabet and number-name graphs for inverse normalization.

NeMo inverts its TN number grammar. This package is standalone, so it builds
the same digit -> leaves -> words graphs from its own copies of the case
lexicons ("1013" -> "1000 13" -> "тысяча тринадцать", every word of one
number in one grammatical case) and inverts them.
"""

from collections import defaultdict
from functools import lru_cache

import pynini
from pynini.lib import byte, pynutil

from itn.utils import get_abs_path, load_labels

RU_LOWER = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
RU_ALPHA = pynini.union(*(RU_LOWER + RU_LOWER.upper())).optimize()
RU_ALPHA_OR_SPACE = (RU_ALPHA | " ").optimize()

DIGIT = byte.DIGIT
NONZERO = pynini.difference(DIGIT, "0").optimize()
CASES = ("1_cardinals_nominative", "2_cardinals_genitive", "3_cardinals_dative",
         "4_cardinals_accusative", "5_cardinals_instrumental", "6_cardinals_prepositional")
SCALES = ("1000", "1000000", "1000000000")


def data_path(name):
    return get_abs_path("russian/data/" + name)


def latin_to_cyrillic():
    return pynini.string_file(data_path("latin_to_cyrillic.tsv")).optimize()


def insertion_labels(path):
    """Label type -> weighted " word" insertions (NeMo's decimal word tables)."""
    mapping = defaultdict(list)
    for key, value, weight in load_labels(path):
        mapping[key].append(pynutil.add_weight(pynutil.insert(" " + value), float(weight)))
    return {key: pynini.union(*values).optimize() for key, values in mapping.items()}


class NumberNames:
    """Digit -> word graphs (``cardinal``, ``nominative``, ``ordinal``) for inversion."""

    def __init__(self):
        factorization = self._factorization()
        ordinal_factorization = factorization | pynutil.add_weight(self._complex_thousands(), -1)

        alternatives = {}
        for _, phrase in load_labels(data_path("numbers/cardinals_alternatives.tsv")):
            alternatives.setdefault(phrase.split()[1], phrase)
        alternative_map = pynini.string_map(sorted(alternatives.items()))

        lexicons = []
        for index, name in enumerate(CASES):
            table = pynini.string_map(load_labels(data_path("numbers/{}.tsv".format(name)))).optimize()
            first = table | (table @ alternative_map)
            if index == 0:
                nominative, nominative_first = table, first
                lexicons.append(pynutil.add_weight(first + pynini.closure(" " + table), -0.1))
            else:
                word = pynutil.add_weight(table, 0.1)
                lexicons.append(pynutil.add_weight(first, 0.1) + pynini.closure(" " + word))
        self.cardinal = (factorization @ pynini.union(*lexicons)).optimize()
        self.nominative = (factorization @ (nominative_first + pynini.closure(" " + nominative))).optimize()

        ordinal_rows = load_labels(data_path("numbers/ordinals.tsv"))
        ordinal_word = pynini.string_map(ordinal_rows).optimize()
        # Spoken input may split compounds: "двух тысячных" for "двухтысячных".
        split = [(row[0], word.replace("тысячн", " тысячн", 1)) + tuple(row[2:])
                 for row in ordinal_rows for word in [row[1]] if row[0] not in SCALES and "тысячн" in word]
        ordinal_word |= pynutil.add_weight(pynini.string_map(split), 1)
        ordinal = ordinal_word | nominative_first + pynini.closure(" " + nominative) + " " + ordinal_word
        self.ordinal = (ordinal_factorization @ ordinal).optimize()

    @staticmethod
    def _two_digits():
        """Exactly two digits with value 1..99 -> leaves."""
        upper = pynini.difference(NONZERO, "1")
        return (
            pynutil.delete("0") + NONZERO
            | pynini.accep("1") + DIGIT
            | upper + pynini.accep("0")
            | upper + pynutil.insert("0 ") + NONZERO
        ).optimize()

    def _three_digits(self):
        """Exactly three digits with value 1..999 -> leaves."""
        two = self._two_digits()
        return (
            NONZERO + pynutil.insert("00") + (pynutil.delete("00") | pynutil.insert(" ") + two)
            | pynutil.delete("0") + two
        ).optimize()

    def _top(self):
        """One to three digits without a leading zero -> leaves."""
        return (NONZERO | (NONZERO + DIGIT) @ self._two_digits()
                | (NONZERO + DIGIT + DIGIT) @ self._three_digits()).optimize()

    def _scaled_top(self, scale):
        """A leading group times a scale; a bare "1" is the scale word alone ("тысяча")."""
        group = pynini.difference(NONZERO + pynini.closure(DIGIT, 0, 2), "1") @ self._top()
        return (group + pynutil.insert(" " + scale) | pynini.cross("1", scale)).optimize()

    def _segment(self, scale):
        group = pynini.difference(DIGIT ** 3, pynini.union("000", "001")) @ self._three_digits()
        group = group + pynutil.insert(" " + scale) | pynini.cross("001", scale)
        return (pynutil.insert(" ") + group | pynutil.delete("000")).optimize()

    def _factorization(self):
        last = pynutil.insert(" ") + self._three_digits() | pynutil.delete("000")
        return (
            pynini.accep("0")
            | self._top()
            | self._scaled_top("1000") + last
            | self._scaled_top("1000000") + self._segment("1000") + last
            | self._scaled_top("1000000000") + self._segment("1000000") + self._segment("1000") + last
        ).optimize()

    def _complex_thousands(self):
        """NeMo's "пятитысячный": a final N x 1000 (N = 2..20) is one ordinal leaf."""
        top = pynini.string_map([(str(n), "{}000".format(n)) for n in range(2, 21)])
        segment = pynutil.insert(" ") + pynini.string_map(
            [("{:03d}".format(n), "{}000".format(n)) for n in range(2, 21)])
        return (
            (top
             | self._scaled_top("1000000") + segment
             | self._scaled_top("1000000000") + self._segment("1000000") + segment)
            + pynutil.delete("000")
        ).optimize()


@lru_cache(maxsize=None)
def number_names():
    return NumberNames()
