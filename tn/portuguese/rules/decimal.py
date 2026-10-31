# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from tn.utils import get_abs_path, load_labels


TOKEN_ORDER = {}


class Decimal(Processor):
    """Comma decimals and quantities as flat raw fields.

    The fractional part is always read digit by digit, e.g. 0,01 -> "zero
    vírgula zero um", and quantities follow the number: 1,5 milhões.
    """

    def __init__(self, cardinal):
        super().__init__("decimal", token_orders=TOKEN_ORDER)
        specials = {row[0]: row[1] for row in load_labels(get_abs_path("portuguese/data/numbers/decimal_specials.tsv"))}
        self.separator = specials["separator"]
        digit = pynini.string_map([(str(n), cardinal.digits[str(n)]) for n in range(1, 10)] + [("0", cardinal.zero)])
        self.integer_graph = cardinal.graph
        self.fractional_graph = (
            pynini.closure(self.DIGIT, 1, 15) @ (digit + (insert(" ") + digit).star)
        ).optimize()
        self.quantity_graph = pynini.string_file(get_abs_path("portuguese/data/numbers/quantity_words.tsv"))
        self.sign = cardinal.sign

        sign_t = (self.tag_field("negative", self.sign) + insert(" ")).ques
        sign_v = (self.verbalize_field("negative", self.sign) + delete(" ")).ques
        quantity_t = self.DELETE_SPACE + insert(" ") + self.tag_field("quantity", self.quantity_graph)
        fraction_t = (
            self.tag_field("integer_part", self.integer_graph) + delete(",") + insert(" ")
            + self.tag_field("fractional_part", self.fractional_graph)
        )
        quantity_integer = pynini.closure(self.DIGIT, 1, 12) @ cardinal.number
        unsigned_t = fraction_t + quantity_t.ques | self.tag_field("integer_part", quantity_integer) + quantity_t
        self.fields_tagger = (sign_t + unsigned_t).optimize()

        quantity_v = insert(" ") + self.verbalize_field("quantity", self.quantity_graph, leading_space=True)
        fraction_v = (
            self.verbalize_field("integer_part", self.integer_graph)
            + insert(" " + self.separator + " ")
            + self.verbalize_field("fractional_part", self.fractional_graph, leading_space=True)
        )
        unsigned_v = fraction_v + quantity_v.ques | self.verbalize_field("integer_part", quantity_integer) + quantity_v
        self.unsigned_verbalizer = unsigned_v.optimize()
        self.unsigned_tagger = unsigned_t.optimize()
        self.fields_verbalizer = (sign_v + unsigned_v).optimize()
        self.tagger = self.add_tokens(self.fields_tagger)
        self.verbalizer = self.delete_tokens(self.fields_verbalizer)
