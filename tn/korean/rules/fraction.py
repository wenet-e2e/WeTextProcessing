# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

# Korean reads the denominator before the numerator ("3/4" -> "사분의 삼"),
# whichever written form ("3/4" or "4분의3") produced the fields.
TOKEN_ORDER = {"fraction": ["negative", "integer_part", "denominator", "numerator", "particle"]}


class Fraction(Processor):
    """Fractions, e.g. ``1과1/2`` -> ``일과 이분의 일``.

    Supports slash (``3/4``) and native (``4분의3``) forms, optional sign,
    mixed integer parts ending in ``과``/``와``, ``√`` operands, decimal
    operands and a trailing subject/topic/object particle whose allomorph is
    chosen from the spoken numerator.
    """

    def __init__(self, cardinal):
        super().__init__("fraction", token_orders=TOKEN_ORDER)
        self.cardinal = cardinal
        self.build_graphs()
        self.build_tagger()
        self.build_verbalizer()

    def build_graphs(self):
        # Bare semantic graphs, also reused by measure.
        number = self.cardinal.graph
        decimal = number + pynini.cross(".", "점") + self.cardinal.digits.star
        self.numeral = (pynini.cross("√", "루트 ").ques + (number | decimal)).optimize()
        self.integer_graph = (self.numeral + pynini.union("과", "와")).optimize()
        self.sign = pynini.string_map([("-", "마이너스 "), ("마이너스", "마이너스 ")]).optimize()
        # Written particles normalize to their post-consonant form; the
        # verbalizer switches to the post-vowel form after the numerator.
        self.particle = pynini.string_map([
            ("이", "이"),
            ("가", "이"),
            ("은", "은"),
            ("는", "은"),
            ("을", "을"),
            ("를", "을"),
        ]).optimize()
        vowel_final = pynini.union("이", "사", "오", "구")
        allomorph = pynini.string_map([("이", "가"), ("은", "는"), ("을", "를")])
        self.particle_agreement = pynini.cdrewrite(allomorph, vowel_final, "[EOS]", self.VSIGMA).optimize()

    def build_tagger(self):
        negative = self.tag_field("negative", self.sign) + insert(" ")
        integer = self.tag_field("integer_part", self.integer_graph) + self.DELETE_SPACE + insert(" ")
        slash = self.tag_field("numerator", self.numeral) + delete("/") + insert(" ")
        slash += self.tag_field("denominator", self.numeral)
        word = self.tag_field("denominator", self.numeral) + delete("분의") + insert(" ")
        word += self.tag_field("numerator", self.numeral)
        particle = insert(" ") + self.tag_field("particle", self.particle)
        tagger = negative.ques + integer.ques + (slash | word) + particle.ques
        self.tagger = self.add_tokens(tagger)

    def build_verbalizer(self):
        negative = self.verbalize_field("negative", self.sign) + delete(" ")
        integer = self.verbalize_field("integer_part", self.integer_graph) + delete(" ") + insert(" ")
        denominator = self.verbalize_field("denominator", self.numeral) + insert("분의 ")
        numerator = self.verbalize_field("numerator", self.numeral, leading_space=True)
        particle = self.verbalize_field("particle", self.particle, leading_space=True)
        numerator |= (numerator + particle) @ self.particle_agreement
        verbalizer = negative.ques + integer.ques + denominator + numerator
        self.verbalizer = self.delete_tokens(verbalizer)
