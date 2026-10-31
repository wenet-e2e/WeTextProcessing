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
    CARDINAL_SEPARATOR,
    DECIMAL_SEPARATOR,
    DELETE_SPACE,
    DIGIT,
    SIGMA,
    WHITE_SPACE,
    add_cardinal_apocope_fem,
    data_file,
    rewrite,
    shift_cardinal_gender,
    shift_number_gender,
    strip_cardinal_apocope,
)

TOKEN_ORDER = {}


class Decimal(Processor):
    """Raw comma decimals and quantities; V reads digit groups and gender."""

    def __init__(self, cardinal, deterministic=True):
        super().__init__("decimal", token_orders=TOKEN_ORDER)
        self.deterministic = deterministic
        insert_space = pynutil.insert(" ")
        zero = cardinal.zero
        digit = cardinal.digit | zero
        if not deterministic:
            graph = pynini.union(digit, cardinal.hundreds, cardinal.tens)
            graph += pynini.closure(insert_space + graph)
        else:
            # Read 1-3 digit groups as cardinals, otherwise tens and digits.
            graph = pynini.union(
                pynutil.add_weight(digit + pynini.closure(insert_space + zero), -0.00001),
                pynutil.add_weight(cardinal.tens + pynini.closure(insert_space + zero), -0.00002),
                pynutil.add_weight(cardinal.hundreds + pynini.closure(insert_space + zero), 0.00001),
                pynutil.add_weight(
                    cardinal.tens + pynini.closure(insert_space + cardinal.tens, 1)
                    + pynini.closure(insert_space + zero, 0, 1)
                    + (pynini.closure(insert_space + digit, 0, 1) | pynini.closure(insert_space + zero)),
                    -0.00002,
                ),
                zero + pynini.closure(insert_space + zero) + pynini.closure(insert_space + digit),
            )
        reverse_apocope = pynini.string_map([("un", "uno"), ("ún", "uno")])
        graph = rewrite(graph, reverse_apocope, right=" ")
        # Digit grouping spaces inside the fraction are not spoken.
        self.fractional_graph = (
            (DIGIT + pynini.closure(pynini.closure(WHITE_SPACE, 0, 1) + DIGIT))
            @ pynini.cdrewrite(DELETE_SPACE, "", "", SIGMA)
            @ graph
        ).optimize()

        stripped = strip_cardinal_apocope(cardinal.graph)
        self.integer_graph = stripped if deterministic else (cardinal.graph | stripped).optimize()
        numbers = pynini.closure(DIGIT, 1, 6) @ cardinal.graph
        self.quantity_integer_graph = (
            pynini.cdrewrite(pynutil.delete(CARDINAL_SEPARATOR), "", "", SIGMA) @ numbers
        ).optimize()
        self.quantity_graph = pynini.project(data_file("numbers/quantities.tsv"), "input").optimize()

        sign_t = cardinal.sign_tagger
        integer_t = self.tag_field("integer_part", self.integer_graph)
        fractional_t = self.tag_field("fractional_part", self.fractional_graph)
        quantity_t = pynini.accep(" ") + self.tag_field("quantity", self.quantity_graph)
        number_t = integer_t + pynutil.delete(DECIMAL_SEPARATOR) + insert_space + fractional_t
        self.unsigned_tagger = (
            number_t
            | self.tag_field("integer_part", self.quantity_integer_graph) + quantity_t
            | number_t + quantity_t
        ).optimize()
        self.fields_tagger = (sign_t + self.unsigned_tagger).optimize()
        self.tagger = self.add_tokens(self.fields_tagger)

        # Verbalization pieces shared with money and measure.
        conjunction = pynutil.insert(" coma ")
        fractional = self.fractional_graph
        if not deterministic:
            conjunction |= pynutil.add_weight(pynutil.insert(pynini.union(" con ", " y ")), 0.001)
            fractional |= pynutil.add_weight(strip_cardinal_apocope(fractional), 0.001)
        sign_v = cardinal.sign_verbalizer
        quantity_v = pynutil.insert(" ") + self.verbalize_field("quantity", self.quantity_graph, leading_space=True)
        integer_v = self.verbalize_field("integer_part", self.integer_graph)
        fractional_v = conjunction + self.verbalize_field("fractional_part", fractional, leading_space=True)
        # Number with a quantity word: always masculine ("millones" is).
        self.quantity_verbalizer = (
            self.verbalize_field("integer_part", self.quantity_integer_graph) + quantity_v
            | integer_v + fractional_v + quantity_v
        ).optimize()
        self.masc_verbalizer = (integer_v + fractional_v).optimize()
        fem_integer = shift_cardinal_gender(self.integer_graph)
        fem_fractional = shift_number_gender(fractional)
        fem = self.verbalize_field("integer_part", fem_integer) + conjunction
        fem += self.verbalize_field("fractional_part", fem_fractional, leading_space=True)
        if not deterministic:
            fem_tail = self.verbalize_field("fractional_part", add_cardinal_apocope_fem(fem_fractional), leading_space=True)
            fem |= pynutil.add_weight(self.verbalize_field("integer_part", fem_integer) + conjunction + fem_tail, 0.001)
        self.fem_verbalizer = fem.optimize()
        self.sign_verbalizer = sign_v
        self.fields_verbalizer = (
            sign_v + (self.masc_verbalizer | self.quantity_verbalizer | pynutil.add_weight(self.fem_verbalizer, 0.001))
        ).optimize()
        self.verbalizer = self.delete_tokens(self.fields_verbalizer)
