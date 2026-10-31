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

from itn.swedish.rules.utils import DIGIT, NONZERO
from tn.processor import Processor

TOKEN_ORDER = {}


class Telephone(Processor):
    """Raw spoken Swedish phone numbers (``noll åtta ...``) to ``08-789 52 25``."""

    def __init__(self, cardinal):
        super().__init__(name="telephone", ordertype="itn", token_orders=TOKEN_ORDER)
        space = pynutil.delete(" ")
        digit = cardinal.digit
        zero = cardinal.zero
        zero_space = zero + space
        tens = cardinal.tens
        hundreds = cardinal.three @ (NONZERO + DIGIT**2)
        two = (tens | zero_space + digit | digit + space + digit).optimize()
        three = (
            hundreds
            | zero_space + tens
            | zero_space + zero_space + digit
            | digit + space + digit + space + digit
            | tens + space + digit
            | digit + space + tens
        ).optimize()
        area = zero_space + (
            digit @ pynini.accep("8")
            | two @ (pynini.union("1", "2", "3", "4", "7") + DIGIT)
            | three @ (pynini.union("5", "6", "9") + DIGIT**2)
        )
        keep = pynini.accep(" ")
        base = pynini.union(
            three + keep + three + keep + two,
            three + keep + two + keep + two,
            two + keep + two + keep + two,
            three + keep + two,
        )
        self.graph = (area + pynini.cross(" ", "-") + base).optimize()
        self.tagger = self.add_tokens(self.tag_field("number_part", self.graph))
        self.verbalizer = self.delete_tokens(self.verbalize_field("number_part", self.graph))
