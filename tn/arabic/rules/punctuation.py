# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from tn.utils import get_abs_path, load_labels


class Punctuation(Processor):
    """Punctuation runs (NeMo's shared TN punctuation grammar), kept verbatim."""

    def __init__(self):
        super().__init__("punctuation", token_orders={})
        excluded = {row[0] for row in load_labels(get_abs_path("arabic/data/punctuation/symbol.tsv"))}
        unicode_marks = [chr(i) for i in range(sys.maxunicode) if category(chr(i)).startswith("P")]
        ascii_marks = list("!#%&'()*+,-./:;<=>?@^_`{|}~\"")
        marks = [mark for mark in dict.fromkeys(unicode_marks + ascii_marks) if mark not in excluded]
        punct = pynini.union(*[pynini.escape(mark) for mark in marks]).optimize()
        whitespace = pynini.union(" ", "\t", "\n", "\r", "\u00a0")
        text = (self.VCHAR - whitespace - pynini.union("<", ">")).plus
        emphasis = pynini.accep("<") + (text + pynini.accep("/").ques | pynini.accep("/") + text) + ">"
        self.graph = (punct.plus | emphasis).optimize()
        # Whitespace written around a mark stays in the raw value.
        space = add_weight(self.SPACE, -0.0001).star
        self.tagger = self.add_tokens(self.tag_field("value", space + self.graph + space))
        graph = self.SPACE.star + self.graph + self.SPACE.star
        self.verbalizer = self.delete_tokens(self.verbalize_field("value", graph))
