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

from itn.marathi.rules.cardinal import MINUS
from tn.processor import Processor


class Decimal(Processor):
    """Spoken decimals, e.g. ``उणे बारा पूर्णांक एक पाच कोटी`` -> ``-१२.१५ कोटी``.

    The tagger keeps the spoken sign, integer, fractional digits and quantity
    and drops the spoken decimal point ``पूर्णांक``.
    """

    def __init__(self, cardinal):
        super().__init__(name="decimal", ordertype="itn", token_orders={})
        digit = (cardinal.digit | cardinal.zero).optimize()
        self.fractional = ((digit + delete(self.SPACE.plus)).star + digit).optimize()
        self.integer = cardinal.graph
        self.quantity_integer = cardinal.hundreds
        self.quantity = pynini.union("लाख", "कोटी", "अब्ज").optimize()
        self.decimal_quantity = (self.quantity | "हजार").optimize()
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        spaces = delete(self.SPACE.plus)
        negative = self.tag_field("negative", pynini.accep(MINUS)) + spaces + insert(" ")
        point = delete("पूर्णांक") + spaces
        integer = self.tag_field("integer_part", self.integer) + spaces + point + insert(" ")
        fractional = self.tag_field("fractional_part", self.fractional)
        decimal = integer + fractional | point + fractional
        quantity = self.tag_field("integer_part", self.quantity_integer) + spaces + insert(" ")
        quantity += self.tag_field("quantity", self.quantity)
        quantity |= decimal + spaces + insert(" ") + self.tag_field("quantity", self.decimal_quantity)
        self.tagger = self.add_tokens(negative.ques + (decimal | quantity))

    def build_verbalizer(self):
        sign = self.verbalize_field("negative", pynini.cross(MINUS, "-")) + delete(" ")
        integer = self.verbalize_field("integer_part", self.integer)
        fractional = insert(".") + self.verbalize_field("fractional_part", self.fractional, leading_space=True)
        decimal = integer + fractional
        decimal |= insert(".") + self.verbalize_field("fractional_part", self.fractional)
        quantity = self.verbalize_field("integer_part", self.quantity_integer) + insert(" ")
        quantity += self.verbalize_field("quantity", self.quantity, leading_space=True)
        quantity |= decimal + insert(" ") + self.verbalize_field("quantity", self.decimal_quantity, leading_space=True)
        self.verbalizer = self.delete_tokens(sign.ques + (decimal | quantity))
