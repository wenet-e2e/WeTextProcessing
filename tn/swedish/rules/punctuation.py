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

import sys
from unicodedata import category

import pynini
from pynini.lib.pynutil import add_weight

from tn.processor import Processor

TOKEN_ORDER = {}

# Symbols that are spoken by other classes rather than kept as punctuation.
_EXCLUDED = set("&#@§™®©_%*+/=^|~$£€₩¥°º")


class Punctuation(Processor):
    def __init__(self, deterministic=True):
        super().__init__("punctuation", token_orders=TOKEN_ORDER)
        self.deterministic = deterministic
        unicode_marks = [chr(i) for i in range(sys.maxunicode)
                         if category(chr(i)).startswith("P") and chr(i) not in ("[", "]")]
        ascii_marks = list("!#%&'()*+,-./:;<=>?@^_`{|}~\"")
        marks = [mark for mark in unicode_marks + ["[", "]"] + ascii_marks if mark not in _EXCLUDED]
        self.punct = pynini.union(*[pynini.escape(mark) for mark in marks]).optimize()
        whitespace = pynini.union(" ", "\t", "\n", "\r", "\u00a0")
        text = (self.VCHAR - whitespace - pynini.union("<", ">")).plus
        emphasis = pynini.accep("<") + (text + pynini.accep("/").ques | pynini.accep("/") + text) + ">"
        self.graph = (self.punct.plus | emphasis).optimize()
        # Whitespace written around a mark stays in the raw value.
        space = add_weight(self.SPACE, -0.0001).star
        self.tagger = self.add_tokens(self.tag_field("value", space + self.graph + space))
        graph = self.SPACE.star + self.graph + self.SPACE.star
        self.verbalizer = self.delete_tokens(self.verbalize_field("value", graph))
