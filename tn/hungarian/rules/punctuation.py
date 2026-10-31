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

import sys
from unicodedata import category

import pynini

from tn.processor import Processor

TOKEN_ORDER = {}

# Symbols that other rules read aloud are not split off as punctuation.
EXCLUDED = set("&#@§™®©_%*+/=^|~$£€₩¥°º")


class Punctuation(Processor):
    """Punctuation marks are kept verbatim."""

    def __init__(self):
        super().__init__("punctuation", token_orders=TOKEN_ORDER)
        unicode_marks = [chr(i) for i in range(sys.maxunicode)
                         if category(chr(i)).startswith("P") and chr(i) not in ("[", "]")]
        ascii_marks = list("!#%&'()*+,-./:;<=>?@^_`{|}~\"")
        self.punct_marks = [mark for mark in unicode_marks + ["[", "]"] + ascii_marks if mark not in EXCLUDED]
        self.punct = pynini.union(*[pynini.escape(mark) for mark in self.punct_marks]).optimize()
        whitespace = pynini.union(" ", "\t", "\n", "\r", "\u00a0")
        text = (self.VCHAR - whitespace - pynini.union("<", ">")).plus
        emphasis = pynini.accep("<") + (text + pynini.accep("/").ques | pynini.accep("/") + text) + ">"
        self.graph = (self.punct.plus | emphasis).optimize()
        self.tagger = self.add_tokens(self.tag_field("value", self.graph))
        self.verbalizer = self.delete_tokens(self.verbalize_field("value", self.graph))
