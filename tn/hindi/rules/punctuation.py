# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import sys
from unicodedata import category

import pynini

from tn.processor import Processor

TOKEN_ORDER = {}


class Punctuation(Processor):
    """Punctuation keeps its written form together with the adjacent spaces.

    The surrounding spaces are part of the raw value, so ``७०, सेक्टर`` keeps the
    input spacing around the comma without a whole-sentence post-processing pass.
    """

    def __init__(self):
        super().__init__("punctuation", token_orders=TOKEN_ORDER)
        marks = [chr(i) for i in range(sys.maxunicode) if category(chr(i)).startswith("P") and chr(i) not in "[]"]
        marks += list("!#%&'()*+,-./:;<=>?@^_`{|}~\"")
        self.punct = pynini.union(*[pynini.escape(mark) for mark in sorted(set(marks))]).optimize()
        text = (self.NOT_SPACE - pynini.union("<", ">")).plus
        emphasis = pynini.accep("<") + (text + pynini.accep("/").ques | pynini.accep("/") + text) + ">"
        self.graph = (self.punct.plus | emphasis).optimize()

        core = self.graph + (self.SPACE.plus + self.graph).star
        spaces = self.SPACE.star
        # Between two tokens, at the start, at the end, and as the whole input.
        self.middle = self.add_tokens(self.tag_field("value", spaces + core + spaces))
        self.leading = self.add_tokens(self.tag_field("value", core + spaces))
        self.trailing = self.add_tokens(self.tag_field("value", spaces + core))
        self.only = self.add_tokens(self.tag_field("value", core))
        self.tagger = self.middle
        self.verbalizer = self.delete_tokens(self.verbalize_field("value", spaces + core + spaces))
