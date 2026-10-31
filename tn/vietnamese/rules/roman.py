# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


class Roman(Processor):
    """``thế kỷ XV`` keeps the keyword and numeral raw and reads "thế kỷ mười lăm"."""

    def __init__(self, cardinal):
        super().__init__("roman", token_orders={})
        keywords = pynini.union(*[row[0] for row in load_labels(get_abs_path("vietnamese/data/roman/key_word.tsv"))])
        numerals = sorted(((int(value), roman) for roman, value in
                           load_labels(get_abs_path("vietnamese/data/roman/roman_numerals.tsv"))), reverse=True)

        def to_roman(number):
            roman = ""
            for value, symbol in numerals:
                while number >= value:
                    roman += symbol
                    number -= value
            return roman

        # Canonical numerals 1..3999, one decimal position at a time (thousands to units).
        positions = []
        for scale in (1000, 100, 10, 1):
            pairs = [(to_roman(digit * scale), str(digit)) for digit in range(1, 10 if scale < 1000 else 4)]
            positions.append(pynini.string_map(pairs) | insert("0"))
        arabic = positions[0]
        for position in positions[1:]:
            arabic += position
        arabic = (arabic @ (delete("0").star + (self.DIGIT - "0") + self.DIGIT.star)).optimize()
        upper = pynini.string_map([(symbol.lower(), symbol) for symbol in "IVXLCDM"])
        arabic |= upper.star @ arabic
        self.graph = (arabic @ cardinal.graph).optimize()
        self.keywords = keywords.optimize()

        self.tagger = self.add_tokens(
            self.tag_field("key", self.keywords) + pynini.accep(" ") + self.tag_field("integer", arabic @ cardinal.raw)
        )
        self.verbalizer = self.delete_tokens(
            self.verbalize_field("key", self.keywords) + pynini.accep(" ")
            + self.verbalize_field("integer", self.graph)
        )
