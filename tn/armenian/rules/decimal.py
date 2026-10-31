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
from pynini.lib.pynutil import delete, insert

from tn.processor import Processor
from tn.utils import get_abs_path


class Decimal(Processor):
    """Decimals and quantities, e.g. ``0.2 միլիարդ`` -> ``զրո ամբողջ երկու միլիարդ``.

    The tagger keeps the written integer and fractional digits (dropping the
    decimal point) and the quantity word; the verbalizer reads the parts
    around ``ամբողջ``.
    """

    def __init__(self, cardinal):
        super().__init__("decimal", token_orders={})
        self.cardinal = cardinal
        self.build_graphs()
        self.build_tagger()
        self.build_verbalizer()

    def build_graphs(self):
        self.integer = self.cardinal.graph
        # NeMo reads fractional parts from 1 to 99 only.
        self.fractional = self.cardinal.one_to_all_tens
        self.quantity = pynini.string_file(get_abs_path("armenian/data/numbers/quantities.tsv")).optimize()
        # Up to six digits before a quantity; group spaces are dropped.
        digits = self.DIGIT + (delete(" ").ques + self.DIGIT).star
        self.quantity_integer = (digits @ pynini.closure(self.DIGIT, 1, 6) @ self.integer).optimize()

    def build_tagger(self):
        point = delete(pynini.union(".", "․")) + insert(" ")
        quantity = delete(" ") + insert(" ") + self.tag_field("quantity", self.quantity)
        fractional = self.tag_field("integer_part", self.integer) + point
        fractional += self.tag_field("fractional_part", self.fractional) + quantity.ques
        whole = self.tag_field("integer_part", self.quantity_integer) + quantity
        self.fields_tagger = (fractional | whole).optimize()
        self.tagger = self.add_tokens(self.fields_tagger)

    def build_verbalizer(self):
        quantity = insert(" ") + self.verbalize_field("quantity", self.quantity, leading_space=True)
        fractional = self.verbalize_field("integer_part", self.integer) + insert(" ամբողջ")
        fractional += insert(" ") + self.verbalize_field("fractional_part", self.fractional, leading_space=True)
        fractional += quantity.ques
        whole = self.verbalize_field("integer_part", self.quantity_integer) + quantity
        self.fields_verbalizer = (fractional | whole).optimize()
        self.verbalizer = self.delete_tokens(self.fields_verbalizer)
