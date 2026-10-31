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
from pynini.lib.pynutil import delete, insert

from tn.hungarian.rules.utils import DIGIT, NONZERO, data_map
from tn.processor import Processor

TOKEN_ORDER = {}

# Long-scale powers above a thousand, lowest first.
POWERS = ["millió", "milliárd", "billió", "billiárd", "trillió", "trilliárd"]


class Cardinal(Processor):
    """Cardinals, e.g. ``2010`` -> ``kettőezer-tíz``, ``-1 000`` -> ``mínusz ezer``.

    The tagger keeps the written digits (with optional ``.``/space group
    separators); the verbalizer reads them. Numbers have up to 24 digits.
    """

    def __init__(self):
        super().__init__("cardinal", token_orders=TOKEN_ORDER)
        self.build_numbers()
        self.sign = pynini.cross("-", "mínusz")
        self.sign_tagger = (self.tag_field("negative", self.sign) + insert(" ")).ques
        self.sign_verbalizer = (self.verbalize_field("negative", self.sign) + pynini.accep(" ")).ques
        self.tagger = self.add_tokens(self.sign_tagger + self.tag_field("integer", self.graph))
        self.verbalizer = self.delete_tokens(self.sign_verbalizer + self.verbalize_field("integer", self.graph))

    def build_numbers(self):
        digit = data_map("number/digit.tsv", invert=True)
        digit_inline = data_map("number/digit_inline.tsv", invert=True)
        tens = data_map("number/tens.tsv", invert=True)
        tens_inline = data_map("number/tens_inline.tsv", invert=True)
        self.zero = data_map("number/zero.tsv", invert=True)
        self.digit = digit
        digit_no_one = (NONZERO - "1") @ digit
        digit_inline_no_one = (NONZERO - "1") @ digit_inline

        graph_tens = (tens_inline + digit | tens).optimize()
        self.tens = graph_tens
        self.two_digit_non_zero = (digit | graph_tens | delete("0") + digit).optimize()
        base_hundreds = pynini.cross("1", "száz") | digit_inline_no_one + insert("száz")
        hundreds = (base_hundreds + (delete("00") | graph_tens | delete("0") + digit)).optimize()
        self.hundreds = hundreds
        # Exactly three digits, not all zero.
        comp = (hundreds | delete("0") + graph_tens | delete("00") + digit).optimize()
        comp_no_one = (hundreds | delete("0") + graph_tens | delete("00") + digit_no_one).optimize()
        # One to three digits without a leading zero.
        head = (digit | graph_tens | hundreds).optimize()
        head_no_one = (digit_no_one | graph_tens | hundreds).optimize()
        self.up_to_hundreds = head

        hyphen = insert("-")
        # Six digits, not all zero.
        level = (
            delete("000") + comp
            | comp_no_one + insert("ezer") + (hyphen + comp | delete("000"))
            | delete("001") + insert("ezer") + (comp | delete("000"))
        ).optimize()
        top = head | (
            head_no_one + insert("ezer") + (hyphen + comp | delete("000"))
            | pynini.cross("1", "ezer") + (comp | delete("000"))
        )
        width = 6
        for word in POWERS:
            rest = hyphen + level | delete("0" * width)
            top |= (pynini.cross("1", word) | head_no_one + insert(word)) + rest
            level = (
                delete("000") + level | (pynini.cross("001", word) | comp_no_one + insert(word)) + rest
            ).optimize()
            width += 3
        self.plain = (self.zero | top).optimize()

        separator = delete(pynini.union(".", " "))
        grouped = NONZERO + pynini.closure(DIGIT, 0, 2) + (separator + DIGIT ** 3).plus
        self.graph = ((NONZERO + DIGIT.star | grouped) @ self.plain | self.zero).optimize()

        zero_space = self.zero + insert(" ")
        self.two_digits_read = ((NONZERO + DIGIT) @ graph_tens | zero_space + digit).optimize()
        self.three_digits_read = (
            hundreds | zero_space + graph_tens | zero_space + zero_space + digit
        ).optimize()
        self.four_digits_read = ((NONZERO + DIGIT ** 3) @ self.plain | zero_space + self.three_digits_read).optimize()
