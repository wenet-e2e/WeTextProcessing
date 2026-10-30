# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pynini
from pynini.lib import pynutil

from tn.processor import Processor


TOKEN_ORDER = {}


class Decimal(Processor):
    """German comma decimals and quantities with reusable flat raw fields."""

    def __init__(self, cardinal, tn_decimal):
        super().__init__("decimal", ordertype="itn", token_orders=TOKEN_ORDER)
        self.fractional_graph = tn_decimal.fractional_graph.invert().optimize()
        self.quantity_graph = tn_decimal.quantity_graph.invert().optimize()
        # 'eine' is a quantity-context spelling, never a replacement in the
        # tagger's raw source value or a general cardinal spelling extension.
        quantity_integer = (
            (self.VSIGMA | pynini.cross("eine", "eins")) @ cardinal.hundreds
        ).optimize()
        quantity_tagger = pynutil.delete(" ") + pynutil.insert(" ") + self.tag_field(
            "quantity", self.quantity_graph,
        )
        decimal_tagger = (
            self.tag_field("integer_part", cardinal.graph_no_exception)
            + pynutil.delete(" komma ") + pynutil.insert(" ")
            + self.tag_field("fractional_part", self.fractional_graph)
        )
        unsigned_tagger = (
            decimal_tagger + pynini.closure(quantity_tagger, 0, 1)
            | self.tag_field("integer_part", quantity_integer) + quantity_tagger
        )
        sign_tagger = pynini.closure(
            self.tag_field("negative", cardinal.sign) + pynutil.delete(" ") + pynutil.insert(" "), 0, 1,
        )
        self.fields_tagger = (sign_tagger + unsigned_tagger).optimize()

        quantity_verbalizer = pynutil.insert(" ") + self.verbalize_field(
            "quantity", self.quantity_graph, leading_space=True,
        )
        decimal_verbalizer = (
            self.verbalize_field("integer_part", cardinal.graph_no_exception)
            + pynutil.insert(",") + self.verbalize_field("fractional_part", self.fractional_graph, leading_space=True)
            + pynini.closure(quantity_verbalizer, 0, 1)
        )
        unsigned_verbalizer = (
            decimal_verbalizer
            | self.verbalize_field("integer_part", quantity_integer) + quantity_verbalizer
        )
        sign_verbalizer = pynini.closure(
            self.verbalize_field("negative", cardinal.sign) + pynutil.delete(" "), 0, 1,
        )
        self.fields_verbalizer = (sign_verbalizer + unsigned_verbalizer).optimize()
        self.tagger = self.add_tokens(self.fields_tagger)
        self.verbalizer = self.delete_tokens(self.fields_verbalizer)
