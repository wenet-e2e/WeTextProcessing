# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


TOKEN_ORDER = {"decimal": ["negative", "integer_part", "fractional_part", "quantity"]}


class Decimal(Processor):
    """``-11,406 millions`` keeps ``-``, ``11``, ``406`` and ``millions`` verbatim."""

    def __init__(self, cardinal):
        super().__init__("decimal", token_orders=TOKEN_ORDER)
        space = insert(" ")
        zero = pynini.string_file(get_abs_path("french/data/numbers/zero.tsv"))
        digit = cardinal.digits
        quantities = pynini.string_file(get_abs_path("french/data/numbers/quantities.tsv"))
        zeros = (space + zero).star
        double = cardinal.double_digits
        # Fractional digits read as single digits, pairs or hundreds (NeMo deterministic weights).
        fractional = pynini.union(
            add_weight(digit + zeros, -0.00001),
            add_weight(double + zeros, -0.00002),
            add_weight(cardinal.hundreds + zeros, 0.00001),
            add_weight(double + (space + double).plus + (space + zero).ques + ((space + digit).ques | zeros), -0.00002),
            zero + zeros + (space + digit).star,
        ).optimize()
        # Before a bare quantity the integer has at most six digits; spaces are dropped.
        compact = self.DIGIT + (delete(" ").ques + self.DIGIT).star
        short_integer = (compact @ pynini.closure(self.DIGIT, 1, 6) @ cardinal.graph).optimize()

        sign = self.tag_field("negative", cardinal.sign) + insert(" ")
        quantity = delete(" ") + insert(" ") + self.tag_field("quantity", quantities)
        number = self.tag_field("integer_part", cardinal.graph) + delete(",") + insert(" ")
        number += self.tag_field("fractional_part", fractional)
        tagger = number + quantity.ques | self.tag_field("integer_part", short_integer) + quantity
        self.tagger = self.add_tokens(sign.ques + tagger)

        sign = self.verbalize_field("negative", cardinal.sign) + pynini.accep(" ")
        quantity = self.verbalize_field("quantity", quantities, leading_space=True)
        number = self.verbalize_field("integer_part", cardinal.graph) + insert(" virgule ")
        number += self.verbalize_field("fractional_part", fractional, leading_space=True)
        verbalizer = number + (insert(" ") + quantity).ques
        verbalizer |= self.verbalize_field("integer_part", short_integer) + insert(" ") + quantity
        self.verbalizer = self.delete_tokens(sign.ques + verbalizer)
