# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pynini.lib import pynutil

from itn.spanish.rules.utils import DELETE_SPACE, cased
from tn.processor import Processor

TOKEN_ORDER = {"decimal": ["negative", "integer_part", "point", "fractional_part", "quantity"]}

QUANTITIES = [
    "millón", "millones", "millardo", "millardos", "billón", "billones",
    "trillón", "trillones", "cuatrillón", "cuatrillones",
]


class Decimal(Processor):
    """Raw spoken decimals; "coma" or "punto" picks the written separator in V."""

    def __init__(self, cardinal, cased_input=False):
        super().__init__("decimal", ordertype="itn", token_orders=TOKEN_ORDER)
        # After the separator: any sequence of numbers below 1000.
        fractional = cardinal.numbers_up_to_thousand
        self.fractional_graph = (pynini.closure(fractional + DELETE_SPACE) + fractional).optimize()
        self.point_graph = cased(pynini.cross("coma", ",") | pynini.cross("punto", "."), cased_input)
        self.integer_graph = cardinal.graph_no_exception
        quantity = pynini.union(*QUANTITIES)
        if cased_input:
            quantity |= pynini.union(*[word[0].upper() + word[1:] for word in QUANTITIES])
        self.quantity_graph = quantity.optimize()

        space = pynini.accep(" ")
        integer_t = self.tag_field("integer_part", self.integer_graph)
        point_t = self.tag_field("point", self.point_graph)
        fractional_t = self.tag_field("fractional_part", self.fractional_graph)
        number_t = pynini.closure(integer_t + space, 0, 1) + point_t + space + fractional_t
        quantity_t = (
            self.tag_field("integer_part", cardinal.numbers_up_to_million) | number_t
        ) + space + self.tag_field("quantity", self.quantity_graph)
        self.unsigned_tagger = (number_t | quantity_t).optimize()
        self.tagger = self.add_tokens(cardinal.sign_tagger + self.unsigned_tagger)

        integer_v = self.verbalize_field("integer_part", self.integer_graph)
        point_v = self.verbalize_field("point", self.point_graph)
        fractional_v = self.verbalize_field("fractional_part", self.fractional_graph, leading_space=True)
        number_v = pynini.closure(integer_v + pynutil.delete(" "), 0, 1) + point_v + fractional_v
        quantity_v = (
            self.verbalize_field("integer_part", cardinal.numbers_up_to_million) | number_v
        ) + pynutil.insert(" ") + self.verbalize_field("quantity", self.quantity_graph, leading_space=True)
        self.unsigned_verbalizer = (number_v | quantity_v).optimize()
        self.verbalizer = self.delete_tokens(cardinal.sign_verbalizer + self.unsigned_verbalizer)
