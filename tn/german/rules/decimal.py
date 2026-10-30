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
from tn.utils import get_abs_path


TOKEN_ORDER = {}


class Decimal(Processor):
    """Comma decimals and quantities, exposing reusable flat written fields."""

    def __init__(self, cardinal, deterministic=True):
        super().__init__("decimal", token_orders=TOKEN_ORDER)
        self.deterministic = deterministic
        self.integer_graph = cardinal.graph
        digit = (
            pynini.string_file(get_abs_path("german/data/numbers/digit.tsv")).invert()
            | pynini.string_file(get_abs_path("german/data/numbers/zero.tsv")).invert()
            | pynini.cross("1", "eins")
        ).optimize()
        self.fractional_graph = (digit + pynini.closure(pynutil.insert(" ") + digit)).optimize()
        self.quantity_graph = pynini.string_file(get_abs_path("german/data/numbers/quantities.tsv")).optimize()

        quantity_tagger = pynini.accep(" ") + self.tag_field("quantity", self.quantity_graph)
        decimal_tagger = (
            self.tag_field("integer_part", self.integer_graph) + pynutil.delete(",") + pynutil.insert(" ")
            + self.tag_field("fractional_part", self.fractional_graph)
        )
        unsigned_tagger = (
            decimal_tagger + pynini.closure(quantity_tagger, 0, 1)
            | self.tag_field("integer_part", cardinal.hundreds) + quantity_tagger
        )
        sign_tagger = pynini.closure(self.tag_field("negative", cardinal.sign) + pynutil.insert(" "), 0, 1)
        self.fields_tagger = (sign_tagger + unsigned_tagger).optimize()

        quantity_verbalizer = pynutil.insert(" ") + self.verbalize_field(
            "quantity", self.quantity_graph, leading_space=True,
        )
        integer_verbalizer = self.verbalize_field("integer_part", self.integer_graph)
        fractional_verbalizer = pynutil.insert(" komma ") + self.verbalize_field(
            "fractional_part", self.fractional_graph, leading_space=True,
        )
        unsigned_verbalizer = integer_verbalizer + (
            quantity_verbalizer | fractional_verbalizer + pynini.closure(quantity_verbalizer, 0, 1)
        )
        sign_verbalizer = pynini.closure(
            self.verbalize_field("negative", cardinal.sign) + pynutil.delete(" "), 0, 1,
        )
        self.fields_verbalizer = (sign_verbalizer + unsigned_verbalizer).optimize()
        self.tagger = self.add_tokens(self.fields_tagger)
        self.verbalizer = self.delete_tokens(self.fields_verbalizer)

        unsigned_number = (
            self.integer_graph + pynini.cross(",", " komma ") + self.fractional_graph
            + pynini.closure(pynini.accep(" ") + self.quantity_graph, 0, 1)
            | cardinal.hundreds + pynini.accep(" ") + self.quantity_graph
        )
        self.number_graph = (pynini.closure(cardinal.sign, 0, 1) + unsigned_number).optimize()
