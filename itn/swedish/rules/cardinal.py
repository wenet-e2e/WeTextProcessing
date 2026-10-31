# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pynini.lib import pynutil

from itn.swedish.rules.utils import DIGIT, NONZERO, SCALES, SPACE, old_spelling, rows, spoken_table, strip_zeros
from tn.processor import Processor

TOKEN_ORDER = {}


class Cardinal(Processor):
    """Raw spoken Swedish integers (with optional inner spaces) to digits."""

    def __init__(self):
        super().__init__(name="cardinal", ordertype="itn", token_orders=TOKEN_ORDER)
        self.build_graphs()
        self.negative = pynini.cross("minus", "-")
        integer = (pynini.difference(self.input_projection(self.graph), pynini.union("en", "ett")) @ self.graph)
        self.no_ones = integer.optimize()
        sign_t = (self.tag_field("negative", self.negative) + pynini.accep(" ")).ques
        sign_v = (self.verbalize_field("negative", self.negative) + pynutil.delete(" ")).ques
        self.tagger = self.add_tokens(sign_t + self.tag_field("integer", self.no_ones))
        self.verbalizer = self.delete_tokens(sign_v + self.verbalize_field("integer", self.graph))

    def build_graphs(self):
        ties_rows = rows("numbers/ties")
        ties = spoken_table("numbers/ties")
        alternatives = []
        for word, value in ties_rows:
            if word.endswith("go"):
                alternatives.append((word[:-2] + "gi", value))
            if word.endswith("tio"):
                alternatives.append((word[:-1], value))
        alternatives += [("förtio", "4"), ("förti", "4"), ("tju", "2")]
        ties = (ties | pynini.string_map(alternatives)).optimize()
        digit = (spoken_table("numbers/digit") | pynini.cross("en", "1")).optimize()
        teen = (spoken_table("numbers/teen") | pynini.cross("aderton", "18")).optimize()
        zero = spoken_table("numbers/zero")
        self.digit = digit
        self.zero = zero
        self.ties = ties
        self.teen = teen

        # Exactly two digits for 10..99, and 01..99.
        self.tens = (ties + SPACE + digit | ties + pynutil.insert("0") | teen).optimize()
        self.below_hundred = (self.tens | pynutil.insert("0") + digit).optimize()
        hundred = (digit + SPACE | pynutil.insert("1")) + pynutil.delete("hundra")
        self.hundred_prefix = hundred.optimize()
        # Exactly three digits for 001..999.
        self.three = (
            hundred + (SPACE + self.below_hundred | pynutil.insert("00"))
            | pynutil.insert("0") + self.below_hundred
        ).optimize()
        thousand = (self.three + SPACE | pynutil.insert("001")) + pynutil.delete("tusen")
        self.thousand_prefix = thousand.optimize()
        self.six = (
            thousand + (SPACE + self.three | pynutil.insert("000"))
            | pynutil.insert("000") + self.three
        ).optimize()
        groups = pynini.accep("")
        self.scale_groups = []
        for scale in SCALES:
            forms = pynini.union(scale, scale + "er", old_spelling(scale), old_spelling(scale) + "er")
            group = (self.three + SPACE | pynutil.insert("001")) + pynutil.delete(forms) + SPACE
            group = (group | pynutil.insert("000")).optimize()
            self.scale_groups.append(group)
            groups += group
        self.groups = groups.optimize()
        number = groups + (self.six | pynutil.insert("000000"))
        self.number = number.optimize()
        positive = strip_zeros() @ (NONZERO + DIGIT.star)
        self.graph = (number @ positive | zero).optimize()
        # 1..999 without leading zeros.
        self.up_to_thousand = (self.three @ positive).optimize()
