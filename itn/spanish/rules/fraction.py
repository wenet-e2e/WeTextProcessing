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

from itn.spanish.rules.utils import SIGMA, cased, data_file, restrict
from tn.processor import Processor

TOKEN_ORDER = {"fraction": ["negative", "integer_part", "numerator", "denominator"]}


class Fraction(Processor):
    """Raw spoken fractions with denominators up to 999; V writes n/d.

    "cuarto(s)" ("room") and fractions without a numerator ("medio dormido")
    are not converted.
    """

    def __init__(self, cardinal, ordinal, cased_input=False):
        super().__init__("fraction", ordertype="itn", token_orders=TOKEN_ORDER)
        singular = pynini.closure(pynutil.delete("s"), 0, 1)

        # Denominators read as ordinals: "un quinto" is 1/5.
        replacements = pynini.invert(data_file("fractions/ordinal_exceptions.tsv"))
        ordinal_graph = ordinal.masc | (replacements @ ordinal.masc)
        ordinal_graph = restrict(ordinal_graph, pynini.union("segundo", "tercero", "cuarto"))
        read_as_ordinals = data_file("fractions/numbers_read_as_ordinals.tsv")
        ordinals = ((SIGMA + singular) @ ordinal_graph @ read_as_ordinals).optimize()
        # Other denominators: "dieciochoavo", "cuarentiunavo".
        accents = pynini.string_map([("un", "ún"), ("dos", "dós"), ("tres", "trés")])
        accents = SIGMA + pynini.closure(pynutil.add_weight(accents, -0.001), 0, 1)
        endings = SIGMA + (pynutil.delete("avo") | pynutil.delete("vo")) + singular
        extend = SIGMA + pynini.closure(pynini.cross("i", "a y "), 0, 1) + SIGMA
        others = extend @ (endings @ (accents @ cardinal.words).optimize()).optimize()
        self.denominator_graph = cased((ordinals | others).optimize(), cased_input)
        self.medio_graph = cased(pynini.cross("medio", "1/2"), cased_input)

        numbers = cardinal.graph_no_exception
        space = pynini.accep(" ")
        sign_t = cardinal.sign_tagger
        integer_t = (
            self.tag_field("integer_part", numbers) + pynutil.delete(cased(pynini.union(" y", " con"), cased_input))
            + space
        )
        fraction_t = self.tag_field("numerator", numbers) + space + self.tag_field("denominator", self.denominator_graph)
        medio_t = self.tag_field("denominator", self.medio_graph)
        self.fields_tagger = (sign_t + pynini.closure(integer_t, 0, 1) + fraction_t).optimize()
        # For measures: also "medio metro" and "tres y medio metros".
        self.proper_tagger = (fraction_t | medio_t).optimize()
        self.mixed_tagger = (self.fields_tagger | sign_t + integer_t + medio_t).optimize()
        self.tagger = self.add_tokens(self.fields_tagger)

        integer_v = self.verbalize_field("integer_part", numbers) + pynini.accep(" ")
        fraction_v = (
            self.verbalize_field("numerator", numbers)
            + pynutil.delete(" ") + pynutil.insert("/")
            + self.verbalize_field("denominator", self.denominator_graph)
        )
        medio_v = self.verbalize_field("denominator", self.medio_graph)
        self.proper_verbalizer = (fraction_v | medio_v).optimize()
        self.fields_verbalizer = (
            cardinal.sign_verbalizer + pynini.closure(integer_v, 0, 1) + self.proper_verbalizer
        ).optimize()
        self.verbalizer = self.delete_tokens(self.fields_verbalizer)
