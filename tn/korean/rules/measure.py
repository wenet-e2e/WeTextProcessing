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
from tn.utils import get_abs_path

# Fraction quantities read the denominator first; units always come last.
TOKEN_ORDER = {
    "measure": [
        "negative", "integer_part", "denominator", "numerator", "particle", "integer", "fractional_part", "units"
    ]
}


class Measure(Processor):
    """Quantities with units, e.g. ``60km/h`` -> ``육십 킬로미터 퍼 시간``.

    The quantity is a cardinal, decimal or fraction (flat fields reusing the
    bare semantic graphs of those rules) followed by a unit or a ``/`` rate.
    """

    def __init__(self, cardinal, decimal, fraction):
        super().__init__("measure", token_orders=TOKEN_ORDER)
        self.cardinal = cardinal
        self.decimal = decimal
        self.fraction = fraction
        self.build_graphs()
        self.build_tagger()
        self.build_verbalizer()

    def build_graphs(self):
        unit = pynini.string_file(get_abs_path("korean/data/measure/unit.tsv"))
        per = pynini.cross("/", "퍼") + self.DELETE_SPACE + insert(" ") + unit
        self.units = (unit + (self.DELETE_SPACE + insert(" ") + per).ques | per).optimize()

    def build_tagger(self):
        decimal = self.decimal
        fraction = self.fraction
        negative = self.tag_field("negative", decimal.sign) + insert(" ") + self.DELETE_SPACE
        cardinal = self.tag_field("integer", self.cardinal.graph)
        decimal = self.tag_field("integer_part", decimal.integer_graph) + delete(".") + insert(" ")
        decimal += self.tag_field("fractional_part", self.decimal.fractional_graph)
        integer = self.tag_field("integer_part", fraction.integer_graph) + self.DELETE_SPACE + insert(" ")
        slash = self.tag_field("numerator", fraction.numeral) + delete("/") + insert(" ")
        slash += self.tag_field("denominator", fraction.numeral)
        word = self.tag_field("denominator", fraction.numeral) + delete("분의") + insert(" ")
        word += self.tag_field("numerator", fraction.numeral)
        particle = insert(" ") + self.tag_field("particle", fraction.particle)
        quantity = cardinal | decimal | integer.ques + (slash | word) + particle.ques
        units = self.DELETE_SPACE + insert(" ") + self.tag_field("units", self.units)
        self.tagger = self.add_tokens(negative.ques + quantity + units)

    def build_verbalizer(self):
        decimal = self.decimal
        fraction = self.fraction
        negative = self.verbalize_field("negative", decimal.sign) + delete(" ")
        cardinal = self.verbalize_field("integer", self.cardinal.graph)
        decimal = self.verbalize_field("integer_part", decimal.integer_graph) + insert("점")
        decimal += self.verbalize_field("fractional_part", self.decimal.fractional_graph, leading_space=True)
        integer = self.verbalize_field("integer_part", fraction.integer_graph) + delete(" ") + insert(" ")
        ratio = self.verbalize_field("denominator", fraction.numeral) + insert("분의 ")
        numerator = self.verbalize_field("numerator", fraction.numeral, leading_space=True)
        particle = self.verbalize_field("particle", fraction.particle, leading_space=True)
        numerator |= (numerator + particle) @ fraction.particle_agreement
        ratio += numerator
        quantity = cardinal | decimal | integer.ques + ratio
        units = insert(" ") + self.verbalize_field("units", self.units, leading_space=True)
        self.verbalizer = self.delete_tokens(negative.ques + quantity + units)
