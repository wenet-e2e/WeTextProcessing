# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from tn.utils import get_abs_path


class Measure(Processor):
    """``12kg``, ``8,5m2``, ``3/4m`` and ``120km/h`` keep number and unit symbols raw."""

    def __init__(self, cardinal, decimal, fraction):
        super().__init__("measure", token_orders={})
        prefixes = pynini.string_file(get_abs_path("vietnamese/data/measure/prefixes.tsv"))
        bases = pynini.string_file(get_abs_path("vietnamese/data/measure/base_units.tsv"))
        special = pynini.string_file(get_abs_path("vietnamese/data/measure/measurements_minimal.tsv"))
        composed = (prefixes + insert(" ") + bases | bases).optimize()
        # Listed symbols win over prefix composition ("mb" is "mê ga bai", not "mi li bai").
        composed = pynini.difference(self.input_projection(composed), self.input_projection(special)) @ composed
        unit = (special | composed).optimize()
        self.unit_graph = (unit + (pynini.cross("/", " trên ") + unit).ques).optimize()

        number = fraction.number_graph
        sep = insert(" ")
        integer = self.tag_field("integer", fraction.number_raw)
        decimal_number = self.tag_field("integer_part", fraction.number_raw) + delete(",") + sep
        decimal_number += self.tag_field("fractional_part", self.DIGIT.plus)
        fraction_number = self.tag_field("numerator", fraction.number_raw) + delete("/") + sep
        fraction_number += self.tag_field("denominator", fraction.number_raw)
        units = delete(" ").ques + sep + self.tag_field("units", self.unit_graph)
        self.fields_tagger = (
            cardinal.negative_tagger.ques + (integer | decimal_number | fraction_number) + units
        ).optimize()
        self.tagger = self.add_tokens(self.fields_tagger)

        number_v = pynini.union(
            self.verbalize_field("integer", number),
            decimal.with_integer_verbalizer,
            fraction.fields_verbalizer,
        )
        units_v = insert(" ") + self.verbalize_field("units", self.unit_graph, True)
        self.fields_verbalizer = (cardinal.negative_verbalizer.ques + number_v + units_v).optimize()
        self.verbalizer = self.delete_tokens(self.fields_verbalizer)
