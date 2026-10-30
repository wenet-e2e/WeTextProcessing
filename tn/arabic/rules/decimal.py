# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from tn.processor import Processor
from tn.utils import get_abs_path


class Decimal(Processor):
    """Decimals, e.g. ``6.88`` -> ``ستة وثمانية وثمانون من مئة``.

    The fractional digits are read as a cardinal over 10, 100, 1000 or 10000.
    The tagger keeps the written digits; ``.`` and ``,`` separators are dropped.
    """

    def __init__(self, cardinal):
        super().__init__("decimal", token_orders={})
        self.cardinal = cardinal
        self.build_graphs()
        self.build_tagger()
        self.build_verbalizer()

    def build_graphs(self):
        number = self.cardinal.graph_with_leading_zeros
        denominators = ["عشرة", "مئة", "ألف", "عشرة آلاف"]
        fractional = pynini.union(*[
            self.DIGIT**width @ number + insert(" من " + denominator)
            for width, denominator in enumerate(denominators, 1)
        ])
        self.sign = self.cardinal.sign
        self.integer_graph = self.cardinal.graph
        self.fractional_graph = (insert("و") + fractional).optimize()
        self.quantity = pynini.string_file(get_abs_path("arabic/data/number/quantities.tsv")).optimize()

    def tag_number(self):
        """Raw decimal fields without the sign, shared with measure."""

        tagger = self.tag_field("integer_part", self.integer_graph)
        tagger += delete(pynini.union(".", ",")) + insert(" ")
        tagger += self.tag_field("fractional_part", self.fractional_graph)
        space = add_weight(delete(" "), -0.1).ques + insert(" ")
        return tagger + (space + self.tag_field("quantity", self.quantity)).ques

    def verbalize_number(self):
        verbalizer = self.verbalize_field("integer_part", self.integer_graph) + insert(" ")
        verbalizer += self.verbalize_field("fractional_part", self.fractional_graph, leading_space=True)
        return verbalizer + (insert(" ") + self.verbalize_field("quantity", self.quantity, leading_space=True)).ques

    def build_tagger(self):
        negative = self.tag_field("negative", self.sign) + insert(" ")
        self.tagger = self.add_tokens(negative.ques + self.tag_number())

    def build_verbalizer(self):
        negative = self.verbalize_field("negative", self.sign) + delete(" ")
        self.verbalizer = self.delete_tokens(negative.ques + self.verbalize_number())
