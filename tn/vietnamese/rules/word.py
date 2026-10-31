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
from pynini.lib.pynutil import add_weight, delete

from tn.processor import Processor


class Word(Processor):
    """Pass-through words; digits and currency/measure symbols start other tokens."""

    def __init__(self):
        super().__init__("word", token_orders={})
        symbols = pynini.union("°", "′", "″", "$", "€", "₩", "£", "¥", "#", "%", "₫", self.DIGIT)
        word = (self.NOT_SPACE - symbols).plus
        # Plain ASCII words are preferred, as in NeMo.
        self.graph = (add_weight(self.ALPHA.plus, -1.0) | word).optimize()
        # Any other non-space character, so that no input is left without a path.
        self.fallback = add_weight(self.NOT_SPACE, 1000).optimize()
        self.tagger = self.add_tokens(self.tag_field("value", self.graph))
        self.fallback_tagger = self.add_tokens(self.tag_field("value", self.fallback))
        value = self.NOT_SPACE.plus | delete(self.SPACE.plus)
        self.verbalizer = self.delete_tokens(self.verbalize_field("value", value))
