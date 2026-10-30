# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
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
from pynini.lib.pynutil import add_weight, delete, insert

from itn.utils import get_abs_path
from itn.vietnamese.rules.cardinal import DELETE_EXTRA_SPACE, DELETE_SPACE
from tn.processor import Processor


# "hai mét rưỡi" is tagged unit-first but written as 2.5 m.
TOKEN_ORDER = {"measure": ["negative", "integer", "integer_part", "fractional_part", "quantity", "units"]}


class Measure(Processor):
    """``năm mươi sáu phẩy ba ki lô mét vuông`` keeps number and unit words raw and becomes ``56.3 km²``."""

    def __init__(self, cardinal, decimal):
        super().__init__(name="measure", ordertype="itn", token_orders=TOKEN_ORDER)
        ds = DELETE_SPACE
        sep = DELETE_EXTRA_SPACE + insert(" ")
        unit = pynini.string_file(get_abs_path("vietnamese/data/measurements.tsv")).invert().optimize()
        per_unit = insert("/") + delete("trên") + ds + unit
        self.units_graph = (unit | per_unit | add_weight(unit + ds + per_unit, 0.01)).optimize()
        self.half_graph = (cardinal.digit | cardinal.half | cardinal.one | cardinal.four).optimize()

        negative = self.tag_field("negative", cardinal.negative) + sep
        units = sep + self.tag_field("units", self.units_graph)
        integer = self.tag_field("integer", cardinal.graph_no_exception)
        trailing = sep + self.tag_field("fractional_part", self.half_graph)
        self.tagger = self.add_tokens(
            negative.ques + (decimal.fields_tagger + units | integer + units + trailing.ques)
        )

        # Time units and percent are glued to the number; every other unit is spaced.
        glued = pynini.union("s", "ms", "ns", "μs", "h", "min", "%")
        units_graph = self.units_graph @ glued
        spaced_graph = self.units_graph @ pynini.difference(self.NOT_SPACE.plus, glued)
        space = delete(" ")
        units_v = space + (self.verbalize_field("units", units_graph)
                           | insert(" ") + self.verbalize_field("units", spaced_graph))
        fraction_graph = decimal.fractional_graph | decimal.scale_fraction_graph | self.half_graph
        integer_v = self.verbalize_field("integer", cardinal.graph_no_exception)
        integer_v += (insert(".") + space + self.verbalize_field("fractional_part", fraction_graph)).ques
        negative_v = decimal.negative_verbalizer
        self.verbalizer = self.delete_tokens(negative_v.ques + (decimal.fields_verbalizer | integer_v) + units_v)
