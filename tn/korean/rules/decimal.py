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


class Decimal(Processor):
    """Decimal numbers, e.g. ``-1.5`` -> ``마이너스 일점오``.

    The tagger keeps the written sign, integer digits and fractional digits;
    the verbalizer reads the integer with Sino-Korean numerals and the
    fractional part digit by digit after ``점``.
    """

    def __init__(self, cardinal):
        super().__init__("decimal", token_orders={})
        self.cardinal = cardinal
        self.build_graphs()
        self.build_tagger()
        self.build_verbalizer()

    def build_graphs(self):
        # Bare semantic graphs, also reused by measure.
        self.sign = pynini.string_map([("-", "마이너스 "), ("마이너스", "마이너스 ")]).optimize()
        self.integer_graph = self.cardinal.graph
        self.fractional_graph = self.cardinal.digits.plus.optimize()
        self.graph = (self.integer_graph + pynini.cross(".", "점") + self.fractional_graph).optimize()

    def build_tagger(self):
        tagger = self.tag_field("negative", self.sign) + insert(" ")
        tagger = tagger.ques + self.tag_field("integer_part", self.integer_graph)
        tagger += delete(".") + insert(" ") + self.tag_field("fractional_part", self.fractional_graph)
        self.tagger = self.add_tokens(tagger)

    def build_verbalizer(self):
        verbalizer = self.verbalize_field("negative", self.sign) + delete(" ")
        verbalizer = verbalizer.ques + self.verbalize_field("integer_part", self.integer_graph)
        verbalizer += insert("점") + self.verbalize_field("fractional_part", self.fractional_graph, leading_space=True)
        self.verbalizer = self.delete_tokens(verbalizer)
