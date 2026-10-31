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


class Fraction(Processor):
    """``trừ năm phần mười hai`` keeps numerator and denominator words and becomes ``-5/12``."""

    def __init__(self, cardinal):
        super().__init__(name="fraction", ordertype="itn", token_orders={})
        sep = DELETE_EXTRA_SPACE + insert(" ")
        self.numerator_graph = cardinal.graph_no_exception
        self.denominator_graph = (cardinal.graph_no_exception | cardinal.four).optimize()
        numerator = self.tag_field("numerator", self.numerator_graph)
        denominator = self.tag_field("denominator", self.denominator_graph)
        fields = numerator + DELETE_SPACE + delete(pynini.union("phần", "trên", "chia")) + sep + denominator
        negative = self.tag_field("negative", cardinal.negative) + sep
        self.tagger = self.add_tokens(negative.ques + fields)

        negative_v = self.verbalize_field("negative", cardinal.negative) + delete(" ")
        fields_v = self.verbalize_field("numerator", self.numerator_graph) + insert("/")
        fields_v += delete(" ") + self.verbalize_field("denominator", self.denominator_graph)
        self.verbalizer = self.delete_tokens(negative_v.ques + fields_v)
