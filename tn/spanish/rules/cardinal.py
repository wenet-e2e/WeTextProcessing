# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pynini.lib import pynutil

from tn.processor import Processor
from tn.spanish.rules.utils import (
    ALPHA,
    CARDINAL_SEPARATOR,
    DELETE_SPACE,
    DIGIT,
    SIGMA,
    WHITE_SPACE,
    add_cardinal_apocope_fem,
    data_file,
    shift_cardinal_gender,
    strip_cardinal_apocope,
)

TOKEN_ORDER = {}


def filter_punctuation(fst):
    """Accepts plain digit strings or thousands groups split by "." or " "."""
    three = DIGIT ** 3
    cardinal_string = pynini.closure(DIGIT, 1) | (
        pynini.closure(DIGIT, 1, 3) + pynutil.delete(CARDINAL_SEPARATOR)
        + pynini.closure(three + pynutil.delete(CARDINAL_SEPARATOR)) + three
    )
    return (cardinal_string @ fst).optimize()


class Cardinal(Processor):
    """Raw Spanish cardinals; digits become (gendered) words only in V."""

    def __init__(self, deterministic=True):
        super().__init__("cardinal", token_orders=TOKEN_ORDER)
        self.deterministic = deterministic

        def table(name):
            return pynini.invert(data_file("numbers/" + name + ".tsv"))

        zero = table("zero")
        digit = table("digit")
        teen, ties, twenties, hundreds = table("teen"), table("ties"), table("twenties"), table("hundreds")
        insert_space = pynutil.insert(" ")
        digits_no_one = (DIGIT - "1") @ digit

        tens = teen | ties + (pynutil.delete("0") | pynutil.insert(" y ") + digit) | twenties
        self.digit = digit.optimize()
        self.zero = zero.optimize()
        self.tens = tens.optimize()
        self.two_digit_non_zero = pynini.union(digit, tens, pynini.cross("0", " ") + digit).optimize()

        graph_hundreds = hundreds + pynini.union(
            pynutil.delete("00"), insert_space + tens, pynini.cross("0", " ") + digit,
        )
        graph_hundreds |= pynini.cross("100", "cien")
        graph_hundreds |= pynini.cross("1", "ciento") + insert_space + pynini.union(tens, pynutil.delete("0") + digit)
        self.hundreds = graph_hundreds.optimize()

        component = pynini.union(graph_hundreds, pynutil.delete("0") + tens)
        component_nonzero = component | pynutil.delete("00") + digit
        component_nonzero_no_one = component | pynutil.delete("00") + digits_no_one
        thousands_rest = insert_space + component_nonzero | pynutil.delete("000")
        thousands_nonzero = pynini.union(
            pynutil.delete("000") + component_nonzero,
            component_nonzero_no_one + pynutil.insert(" mil") + thousands_rest,
            pynini.cross("001", "mil") + thousands_rest,
        )
        thousands_nonzero_no_one = pynini.union(
            pynutil.delete("000") + component_nonzero_no_one,
            component_nonzero_no_one + pynutil.insert(" mil") + thousands_rest,
            pynini.cross("001", "mil") + thousands_rest,
        )

        def power(singular, plural):
            graph = pynutil.add_weight(pynini.cross("000001", singular), -0.001)
            graph |= thousands_nonzero_no_one + pynutil.insert(plural)
            graph |= pynutil.delete("000000")
            return graph + insert_space

        graph = (
            power("un trillón", " trillones") + power("un billón", " billones") + power("un millón", " millones")
            + (thousands_nonzero | pynutil.delete("000000"))
        )
        graph = (
            ((DIGIT - "0") + pynini.closure(DIGIT))
            @ pynini.cdrewrite(pynini.closure(pynutil.insert("0")), "[BOS]", "", SIGMA)
            @ DIGIT ** 24
            @ graph
            @ pynini.cdrewrite(DELETE_SPACE, "[BOS]", "", SIGMA)
            @ pynini.cdrewrite(DELETE_SPACE, "", "[EOS]", SIGMA)
            @ pynini.cdrewrite(pynini.cross(pynini.closure(WHITE_SPACE, 2), " "), ALPHA, ALPHA, SIGMA)
        )
        # Written digits (optionally grouped) to masculine words with apocope.
        self.graph = filter_punctuation(graph | zero)

        self.masc = self.graph
        self.fem = shift_cardinal_gender(self.graph)
        # Gender is free for a bare cardinal: prefer masculine, keep feminine.
        spoken = self.masc | pynutil.add_weight(self.fem, 0.001)
        if not deterministic:
            spoken |= pynutil.add_weight(strip_cardinal_apocope(self.masc), 0.002)
            spoken |= pynutil.add_weight(add_cardinal_apocope_fem(self.fem), 0.003)
        self.spoken = spoken.optimize()

        self.sign = pynini.cross("-", "menos ")
        self.sign_tagger = pynini.closure(self.tag_field("negative", self.sign) + pynutil.insert(" "), 0, 1)
        self.sign_verbalizer = pynini.closure(self.verbalize_field("negative", self.sign) + pynutil.delete(" "), 0, 1)
        self.tagger = self.add_tokens(self.sign_tagger + self.tag_field("integer", self.graph))
        self.verbalizer = self.delete_tokens(self.sign_verbalizer + self.verbalize_field("integer", self.spoken))
