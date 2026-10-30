# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from unicodedata import category

import pynini

from tn.processor import Processor
from tn.utils import get_abs_path, load_labels


TOKEN_ORDER = {}


class Punctuation(Processor):
    def __init__(self, deterministic=True):
        super().__init__("punctuation", token_orders=TOKEN_ORDER)
        self.deterministic = deterministic
        excluded = {row[0] for row in load_labels(get_abs_path("english/data/whitelist/symbol.tsv"))}
        unicode_marks = [chr(i) for i in range(sys.maxunicode)
                         if category(chr(i)).startswith("P") and chr(i) not in ("[", "]")]
        ascii_marks = list("!#%&'()*+,-./:;<=>?@^_`{|}~\"")
        self.punct_marks = [mark for mark in unicode_marks + ["[", "]"] + ascii_marks if mark not in excluded]
        self.punct = pynini.union(*[pynini.escape(mark) for mark in self.punct_marks]).optimize()
        whitespace = pynini.union(" ", "\t", "\n", "\r", "\u00a0")
        text = (self.VCHAR - whitespace - pynini.union("<", ">")).plus
        emphasis = pynini.accep("<") + (text + pynini.accep("/").ques | pynini.accep("/") + text) + ">"
        self.graph = (self.punct.plus | emphasis).optimize()
        self.tagger = self.add_tokens(self.tag_field("value", self.graph))
        self.verbalizer = self.delete_tokens(self.verbalize_field("value", self.graph))
