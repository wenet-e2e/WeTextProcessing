# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


TOKEN_ORDER = {"decimal": ["negative", "integer_part", "fractional_part", "quantity"]}

WHITE_SPACE = pynini.union(" ", "\t", "\n", "\r", "\u00a0").optimize()


class Decimal(Processor):
    """``moins un virgule deux six`` keeps ``moins``, ``un`` and ``deux six`` verbatim -> ``-1,26``.

    Also covers round amounts such as ``quatre-cent-soixante millions`` -> ``460 millions``.
    """

    def __init__(self, cardinal):
        super().__init__("decimal", ordertype="itn", token_orders=TOKEN_ORDER)
        self.delete_space = delete(WHITE_SPACE.star)
        self.delete_extra_space = delete(WHITE_SPACE.plus) + insert(" ")
        fractional = cardinal.numbers_up_to_million
        fractional = (pynini.closure(fractional + self.delete_space) + fractional).optimize()
        integer = cardinal.graph_no_exception
        strip = delete(pynini.accep("0").star) + (self.DIGIT - "0") + self.DIGIT.star
        round_integer = (cardinal.hundreds_nonzero @ strip).optimize()
        quantity = pynini.union(
            "million", "millions", "milliard", "milliards", "billion", "billions",
            "billiard", "billiards", "trillion", "trillions", "trilliard", "trilliards",
        ).optimize()

        three = self.DIGIT**3
        at_most_three = pynini.closure(self.DIGIT, 1, 3)
        self.group_integer = (at_most_three + (insert(" ") + three).star).optimize()
        group_fractional = ((three + insert(" ")).star + at_most_three).optimize()
        self.sign = cardinal.sign

        # Tagger: separators ("virgule", hyphens, whitespace) are dropped between raw fields.
        integer_tag = self.tag_field("integer_part", integer) + self.delete_extra_space
        number = integer_tag.ques + delete("virgule") + delete(WHITE_SPACE.plus)
        number += self.tag_field("fractional_part", fractional)
        quantity_tag = insert(" ") + self.tag_field("quantity", quantity)
        round_number = self.tag_field("integer_part", round_integer) + (delete("-").ques | delete(WHITE_SPACE.plus))
        self.number_tagger = (number | number + delete(WHITE_SPACE.plus) + quantity_tag | round_number + quantity_tag)
        self.sign_tagger = self.tag_field("negative", self.sign) + self.delete_extra_space
        self.tagger = self.add_tokens(self.sign_tagger.ques + self.number_tagger)

        integer_out = self.verbalize_field("integer_part", integer @ self.group_integer)
        fractional_out = insert(",") + self.verbalize_field("fractional_part", fractional @ group_fractional)
        quantity_out = insert(" ") + self.verbalize_field("quantity", quantity, leading_space=True)
        number = (integer_out + delete(" ")).ques + fractional_out + quantity_out.ques
        round_number = self.verbalize_field("integer_part", round_integer @ self.group_integer) + quantity_out
        self.number_verbalizer = (number | round_number).optimize()
        self.sign_verbalizer = self.verbalize_field("negative", self.sign) + delete(" ")
        self.verbalizer = self.delete_tokens(self.sign_verbalizer.ques + self.number_verbalizer)
