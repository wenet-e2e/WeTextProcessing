# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from tn.italian.rules.utils import load_words
from tn.processor import Processor


class Decimal(Processor):
    """Decimals and quantities, e.g. ``4,56 milioni`` -> ``quattro virgola cinquantasei milioni``.

    The tagger keeps the written integer digits, the fractional digits
    (dropping the decimal comma) and the quantity word; the verbalizer reads
    the parts around ``virgola``.
    """

    def __init__(self, cardinal):
        super().__init__("decimal", token_orders={})
        self.cardinal = cardinal
        self.build_graphs()
        self.build_tagger()
        self.build_verbalizer()

    def build_graphs(self):
        cardinal = self.cardinal
        space = insert(" ")
        zero = pynini.cross("0", "zero")
        digit = cardinal.digits | zero
        zeros = (space + zero).star
        tens = cardinal.two_digits
        # NeMo's deterministic reading of the fractional digits: whole
        # numbers up to the hundreds, then pairs, with explicit zeros.
        fractional = pynini.union(
            add_weight(digit + zeros, -0.00001),
            add_weight(tens + zeros, -0.00002),
            add_weight(cardinal.three_digits + zeros, 0.00001),
            add_weight(
                tens + (space + tens).plus + (space + zero).ques + ((space + digit).ques | zeros),
                -0.00002,
            ),
            zero + zeros + (space + digit).star,
        )
        self.fractional = fractional.optimize()
        self.integer = cardinal.graph
        self.quantity = pynini.union(*load_words("italian/data/numbers/quantities.tsv")).optimize()
        # Up to six digits before a bare quantity; separators are dropped.
        separator = delete(pynini.union(".", " "))
        digits = self.DIGIT + (separator.ques + self.DIGIT).star
        self.quantity_integer = (digits @ pynini.closure(self.DIGIT, 1, 6) @ cardinal.number).optimize()

    def build_tagger(self):
        sign = (self.tag_field("negative", self.cardinal.sign) + insert(" ")).ques
        quantity = delete(" ") + insert(" ") + self.tag_field("quantity", self.quantity)
        fractional = self.tag_field("integer_part", self.integer) + delete(",") + insert(" ")
        fractional += self.tag_field("fractional_part", self.fractional) + quantity.ques
        whole = self.tag_field("integer_part", self.quantity_integer) + quantity
        self.fields_tagger = (fractional | whole).optimize()
        self.tagger = self.add_tokens(sign + self.fields_tagger)

    def build_verbalizer(self):
        sign = (self.verbalize_field("negative", self.cardinal.sign) + delete(" ") + insert(" ")).ques
        quantity = insert(" ") + self.verbalize_field("quantity", self.quantity, leading_space=True)
        fractional = self.verbalize_field("integer_part", self.integer) + insert(" virgola")
        fractional += insert(" ") + self.verbalize_field("fractional_part", self.fractional, leading_space=True)
        fractional += quantity.ques
        whole = self.verbalize_field("integer_part", self.quantity_integer) + quantity
        self.fields_verbalizer = (fractional | whole).optimize()
        self.verbalizer = self.delete_tokens(sign + self.fields_verbalizer)
