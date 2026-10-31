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

from itn.swedish.rules.utils import NONZERO, DIGIT, rows
from tn.processor import Processor

TOKEN_ORDER = {}


class Decimal(Processor):
    """Raw spoken decimals and quantities (``tre komma fem miljon``) to ``3,5 miljoner``."""

    def __init__(self, cardinal):
        super().__init__(name="decimal", ordertype="itn", token_orders=TOKEN_ORDER)
        digit = cardinal.digit | cardinal.zero
        sequence = digit + (pynutil.delete(" ") + digit).star
        self.fractional = (
            cardinal.three @ (NONZERO + DIGIT**2) | cardinal.tens | sequence
        ).optimize()
        one = pynini.cross(pynini.union("en", "ett"), "1")
        integer = cardinal.up_to_thousand
        no_one = (pynini.difference(self.input_projection(integer), pynini.union("en", "ett")) @ integer).optimize()

        quantities = pynini.string_map([(row[0], row[-1]) for row in rows("numbers/millions")])
        plural = quantities + "er" | quantities + pynutil.insert("er")
        self.quantity_singular = quantities.optimize()
        self.quantity_plural = plural.optimize()

        negative = cardinal.negative
        sign_t = (self.tag_field("negative", negative) + pynini.accep(" ")).ques
        sign_v = (self.verbalize_field("negative", negative) + pynutil.delete(" ")).ques
        integer_t = self.tag_field("integer_part", integer)
        fractional_t = pynini.accep(" ") + pynutil.delete("komma ") + self.tag_field("fractional_part", self.fractional)
        separator = pynini.accep(" ") | pynutil.insert(" ")
        quantity_t = separator + self.tag_field("quantity", self.quantity_plural)
        tagger = sign_t + (
            self.tag_field("integer_part", one) + separator + self.tag_field("quantity", self.quantity_singular)
            | self.tag_field("integer_part", no_one) + quantity_t
            | integer_t + fractional_t + quantity_t.ques
        )
        self.tagger = self.add_tokens(tagger)

        def quantity_v(graph):
            return pynutil.delete(" ") + pynutil.insert(" ") + self.verbalize_field("quantity", graph)

        fractional_v = pynutil.delete(" ") + pynutil.insert(",") + self.verbalize_field(
            "fractional_part", self.fractional)
        verbalizer = sign_v + (
            self.verbalize_field("integer_part", one) + quantity_v(self.quantity_singular)
            | self.verbalize_field("integer_part", no_one) + quantity_v(self.quantity_plural)
            | self.verbalize_field("integer_part", integer) + fractional_v + quantity_v(self.quantity_plural).ques
        )
        self.verbalizer = self.delete_tokens(verbalizer)
