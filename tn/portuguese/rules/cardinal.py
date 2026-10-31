# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pynini
from pynini.lib.pynutil import delete, insert

from tn.processor import Processor
from tn.utils import get_abs_path, load_labels


TOKEN_ORDER = {}


def number_table(name):
    return {row[0]: row[1] for row in load_labels(get_abs_path("portuguese/data/numbers/" + name))}


def feminine(word):
    """Feminine form of one masculine cardinal word (um, dois, -centos)."""
    if word == "um":
        return "uma"
    if word == "dois":
        return "duas"
    if word.endswith("entos"):
        return word[:-5] + "entas"
    return word


class Cardinal(Processor):
    """Raw Portuguese cardinal digits; spelling happens only in verbalization.

    Groups of three digits are spelled from the NeMo number tables. Following
    the NeMo fixtures, the connector "e" precedes the last non-zero group
    when it is below one hundred or a round hundred, e.g. 1001000 -> "um
    milhão e mil" but 1001001 -> "um milhão mil e um".
    """

    def __init__(self):
        super().__init__("cardinal", token_orders=TOKEN_ORDER)
        self.digits = number_table("digit.tsv")
        self.teens = number_table("teens.tsv")
        self.tens = number_table("tens.tsv")
        self.hundreds = number_table("hundreds.tsv")
        self.zero = number_table("zero.tsv")["0"]
        specials = number_table("cardinal_specials.tsv")
        self.connector = specials["connector"]
        self.thousand = specials["thousand"]
        self.hundred_100 = specials["hundred_100"]
        self.hundred_1 = specials["hundred_1"]
        rows = load_labels(get_abs_path("portuguese/data/numbers/scales.tsv"))[1:]
        self.scales = [(row[0], row[1].strip()) for row in rows]
        self.scale_words = [word for one, plural in self.scales for word in (one.split()[-1], plural)]

        self.number = self.build_number(False)
        self.number_fem = self.build_number(True)
        three = self.DIGIT**3
        separated = []
        for separator in (".", " "):
            separated.append((self.DIGIT - "0") + pynini.closure(self.DIGIT, 0, 2) + (delete(separator) + three).plus)
        grouping = pynini.union(*separated).optimize()
        self.grouped = (grouping @ self.number).optimize()
        self.graph = (self.number | self.grouped).optimize()
        self.graph_fem = (self.number_fem | grouping @ self.number_fem).optimize()
        self.sign = pynini.cross("-", "menos ")

        sign_tagger = (self.tag_field("negative", self.sign) + insert(" ")).ques
        sign_verbalizer = (self.verbalize_field("negative", self.sign) + delete(" ")).ques
        self.tagger = self.add_tokens(sign_tagger + self.tag_field("integer", self.graph))
        self.verbalizer = self.delete_tokens(sign_verbalizer + self.verbalize_field("integer", self.graph))

    def words(self, value, fem=False):
        """Spells 1..999."""
        gender = feminine if fem else (lambda word: word)
        if value == 100:
            return self.hundred_100
        hundred, rest = divmod(value, 100)
        parts = []
        if hundred:
            parts.append(self.hundred_1 if hundred == 1 else gender(self.hundreds[str(hundred)]))
        if rest:
            if rest < 10:
                parts.append(gender(self.digits[str(rest)]))
            elif rest < 20:
                parts.append(self.teens[str(rest)])
            else:
                ten, unit = divmod(rest, 10)
                word = self.tens[str(ten)]
                if unit:
                    word += " {} {}".format(self.connector, gender(self.digits[str(unit)]))
                parts.append(word)
        return " {} ".format(self.connector).join(parts)

    def groups(self, fem):
        """Returns (pure, compound) graphs over padded three-digit groups.

        Pure groups are below one hundred or round hundreds; only they take
        the connector "e" when they are the last non-zero group.
        """
        two = pynini.string_map([("{:02d}".format(value), self.words(value, fem)) for value in range(1, 100)])
        hundreds = pynini.string_map([(str(value), self.words(100 * value, fem)) for value in range(2, 10)])
        pure = (
            delete("0") + two
            | pynini.cross("100", self.hundred_100)
            | hundreds + delete("00")
        )
        compound = (
            (pynini.cross("1", self.hundred_1) | hundreds)
            + insert(" {} ".format(self.connector)) + two
        )
        return pure.optimize(), compound.optimize()

    def scaled(self, group, scale):
        """Appends the scale word of one non-zero three-digit group."""
        if scale == 0:
            return group
        if scale == 1:
            one, plural = self.thousand, " " + self.thousand
        else:
            one, plural = self.scales[scale - 2]
            plural = " " + plural
        scaled = (pynini.difference(self.DIGIT**3, "001") @ group) + insert(plural)
        # "001" belongs to the pure groups, so it keeps their connector.
        one_group = pynini.accep("001") @ group
        if one_group.num_states():
            scaled |= pynini.cross("001", one)
        return scaled

    def build_number(self, fem):
        connector = " {} ".format(self.connector)
        unpadded = (
            insert("00") + (self.DIGIT - "0")
            | insert("0") + (self.DIGIT - "0") + self.DIGIT
            | (self.DIGIT - "0") + self.DIGIT**2
        )
        # Non-zero tails of k three-digit groups, each starting with its joiner.
        non_zero = None
        graphs = [pynini.cross("0", self.zero)]
        for count in range(len(self.scales) + 2):
            if count:
                scale = count - 1
                # Millions and above are masculine nouns: "duas mil", "dois milhões".
                pure, compound = self.groups(fem and scale < 2)
                lower_zeros = delete("0" * (3 * scale))
                tail = (
                    insert(connector) + self.scaled(pure, scale) + lower_zeros
                    | insert(" ") + self.scaled(compound, scale) + lower_zeros
                )
                if non_zero is not None:
                    tail |= delete("000") + non_zero
                    tail |= insert(" ") + self.scaled(pure | compound, scale) + non_zero
                non_zero = tail.optimize()
            pure, compound = self.groups(fem and count < 2)
            top = (unpadded @ self.scaled(pure | compound, count)).optimize()
            lower = delete("0" * (3 * count))
            if count:
                lower |= non_zero
            graphs.append(top + lower)
        return pynini.union(*graphs).optimize()
