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
from pynini.lib.pynutil import delete, insert

from itn.vietnamese.rules.cardinal import DELETE_EXTRA_SPACE, DELETE_SPACE
from tn.processor import Processor


# Scale-first readings ("một triệu hai") are tagged before their fraction digit.
TOKEN_ORDER = {"decimal": ["negative", "integer_part", "fractional_part", "quantity"]}


class Decimal(Processor):
    """``âm chín chín chấm chín lăm tỷ`` keeps its words raw and becomes ``-99.95 tỷ``."""

    def __init__(self, cardinal):
        super().__init__(name="decimal", ordertype="itn", token_orders=TOKEN_ORDER)
        ds = DELETE_SPACE
        sep = DELETE_EXTRA_SPACE + insert(" ")
        base = cardinal.digit | cardinal.zero
        self.fractional_graph = pynini.union(
            base,
            cardinal.four,
            (base + ds).plus + (base | cardinal.four | cardinal.five | cardinal.one),
        ).optimize()
        # The single digit after a scale word: "một triệu hai" -> 1.2 triệu.
        self.scale_fraction_graph = (cardinal.last_digit | cardinal.half).optimize()
        self.scale_integer_graph = (cardinal.nonzero_component @ cardinal.strip_zeros).optimize()
        magnitudes = pynini.union("triệu", "tỉ", "tỷ", "vạn")
        thousands = pynini.union("ngàn", "nghìn")
        self.magnitude_graph = magnitudes.optimize()

        point = delete(pynini.union("chấm", "phẩy"))
        integer = self.tag_field("integer_part", cardinal.graph_no_exception)
        fraction = self.tag_field("fractional_part", self.fractional_graph)
        plain = (integer + sep).ques + point + sep + fraction
        scaled = self.tag_field("integer_part", self.scale_integer_graph) + sep
        scaled += self.tag_field("quantity", magnitudes)
        scaled += (sep + self.tag_field("fractional_part", self.scale_fraction_graph)).ques
        scaled |= plain + sep + self.tag_field("quantity", magnitudes | thousands)
        self.fields_tagger = (plain | scaled).optimize()
        self.negative_tagger = self.tag_field("negative", cardinal.negative) + sep
        self.tagger = self.add_tokens(self.negative_tagger.ques + self.fields_tagger)

        # Fields arrive in TOKEN_ORDER: [integer] [fraction] [quantity].
        integer_graph = cardinal.graph_no_exception | self.scale_integer_graph
        fraction_graph = self.fractional_graph | self.scale_fraction_graph
        integer_v = self.verbalize_field("integer_part", integer_graph)
        fraction_v = insert(".") + self.verbalize_field("fractional_part", fraction_graph)
        quantity_v = insert(" ") + self.verbalize_field("quantity", magnitudes | thousands)
        space = delete(" ")
        self.fields_verbalizer = pynini.union(
            integer_v + (space + fraction_v).ques + (space + quantity_v).ques,
            fraction_v + (space + quantity_v).ques,
        ).optimize()
        self.negative_verbalizer = self.verbalize_field("negative", cardinal.negative) + delete(" ")
        self.verbalizer = self.delete_tokens(self.negative_verbalizer.ques + self.fields_verbalizer)
