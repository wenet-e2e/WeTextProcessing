# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

from itn.armenian.rules.utils import preferred_map
from tn.processor import Processor


class Decimal(Processor):
    """Spoken decimals, e.g. ``հինգ ու կես միլիոն`` -> ``5.5 միլիոն``.

    The tagger keeps the spoken integer, fractional digits and quantity and
    drops the spoken decimal point (``ամբողջ`` or ``ու``).
    """

    def __init__(self, cardinal):
        super().__init__(name="decimal", ordertype="itn", token_orders={})
        self.cardinal = cardinal
        self.build_graphs()
        self.build_tagger()
        self.build_verbalizer()

    def build_graphs(self):
        digit = preferred_map("armenian/data/numbers/digit.tsv")
        digit |= pynini.string_map([("զրո", "0"), ("կես", "5")])
        self.fractional = ((digit + self.DELETE_SPACE).star + digit).optimize()
        self.integer = self.cardinal.graph
        self.quantity_integer = self.cardinal.hundreds
        self.quantity = pynini.union("միլիոն", "միլիարդ", "տրիլիոն").optimize()
        self.decimal_quantity = (self.quantity | "հազար").optimize()

    def build_tagger(self):
        spaces = delete(self.SPACE.plus)
        point = delete(pynini.union("ամբողջ", "ու")) + spaces
        integer = self.tag_field("integer_part", self.integer) + spaces + point + insert(" ")
        fractional = self.tag_field("fractional_part", self.fractional)
        decimal = (integer + fractional | point + fractional).optimize()
        quantity = self.tag_field("integer_part", self.quantity_integer) + spaces + insert(" ")
        quantity += self.tag_field("quantity", self.quantity)
        quantity |= decimal + spaces + insert(" ") + self.tag_field("quantity", self.decimal_quantity)
        self.fields_tagger = (decimal | quantity).optimize()
        self.tagger = self.add_tokens(self.fields_tagger)

    def build_verbalizer(self):
        integer = self.verbalize_field("integer_part", self.integer)
        fractional = insert(".") + self.verbalize_field("fractional_part", self.fractional, leading_space=True)
        decimal = integer + fractional
        decimal |= insert(".") + self.verbalize_field("fractional_part", self.fractional)
        quantity = self.verbalize_field("integer_part", self.quantity_integer) + insert(" ")
        quantity += self.verbalize_field("quantity", self.quantity, leading_space=True)
        quantity |= decimal + insert(" ") + self.verbalize_field("quantity", self.decimal_quantity, leading_space=True)
        self.fields_verbalizer = (decimal | quantity).optimize()
        self.verbalizer = self.delete_tokens(self.fields_verbalizer)
