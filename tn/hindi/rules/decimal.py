# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from tn.hindi.rules.utils import MINUS
from tn.processor import Processor

TOKEN_ORDER = {}


class Decimal(Processor):
    """``-१२.५`` -> ``decimal { negative: "-" integer_part: "१२" fractional_part: "५" }``."""

    def __init__(self, cardinal):
        super().__init__("decimal", token_orders=TOKEN_ORDER)
        single = cardinal.digit | cardinal.zero
        self.fractional_graph = (single + (insert(" ") + single).star).optimize()
        self.integer_graph = cardinal.graph_without_leading_zeros

        self.number_tagger = (self.tag_field("integer_part", self.integer_graph) + delete(".") + insert(" ")
                              + self.tag_field("fractional_part", self.fractional_graph))
        self.number_verbalizer = (self.verbalize_field("integer_part", self.integer_graph) + insert(" दशमलव ")
                                  + self.verbalize_field("fractional_part", self.fractional_graph, leading_space=True))
        self.sign_tagger = self.tag_field("negative", pynini.accep("-")) + insert(" ")
        self.sign_verbalizer = self.verbalize_field("negative", pynini.cross("-", MINUS)) + delete(" ")

        self.tagger = self.add_tokens(self.sign_tagger.ques + self.number_tagger)
        self.verbalizer = self.delete_tokens(self.sign_verbalizer.ques + self.number_verbalizer)
