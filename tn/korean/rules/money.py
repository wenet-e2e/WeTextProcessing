# Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
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


TOKEN_ORDER = {"money": ["integer_part", "currency_maj", "minor_part"]}


class Money(Processor):
    """Preserve written amounts and currencies; pronounce them after reordering."""

    def __init__(self, cardinal):
        super().__init__("money", token_orders=TOKEN_ORDER)
        nonzero = self.DIGIT - "0"
        plain = "0" | (nonzero + self.DIGIT.star)
        grouped = nonzero + pynini.closure(self.DIGIT, 0, 2)
        grouped += (delete(",") + self.DIGIT**3).plus
        number = ((plain | grouped) @ cardinal.graph).optimize()
        scale = pynini.union("만", "억", "조")
        amount = (number | number + scale).optimize()
        minor = (self.DIGIT**2 @ cardinal.graph).optimize()

        currency_prefix = pynini.string_file(get_abs_path("korean/data/money/currency_major.tsv"))
        currency_suffix = currency_prefix.copy().project("output").optimize()
        currency = (currency_prefix | currency_suffix).optimize()

        integer_tag = self.tag_field("integer_part", number)
        minor_tag = delete(".") + insert(" ") + self.tag_field("minor_part", minor)
        number_tag = integer_tag + minor_tag.ques
        number_tag |= self.tag_field("integer_part", number + scale)
        prefix = self.tag_field("currency_maj", currency_prefix) + self.DELETE_SPACE + insert(" ")
        suffix = insert(" ") + self.tag_field("currency_maj", currency_suffix)
        self.tagger = self.add_tokens(prefix + number_tag | number_tag + suffix)

        integer = self.verbalize_field("integer_part", amount)
        won = currency @ pynini.accep("원")
        other = currency @ pynini.difference(self.VSIGMA, "원")
        won_reading = integer + self.verbalize_field("currency_maj", won, leading_space=True)
        other_reading = integer + insert(" ")
        other_reading += self.verbalize_field("currency_maj", other, leading_space=True)
        # NeMo intentionally discards the two-digit minor amount for all currencies.
        minor_drop = self.verbalize_field("minor_part", delete(self.input_projection(minor)), leading_space=True)
        self.verbalizer = self.delete_tokens(
            (won_reading | add_weight(other_reading, 0.5)) + minor_drop.ques)
