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
from pynini.lib import pynutil

from tn.processor import Processor

TOKEN_ORDER = {}


class Word(Processor):
    """Any other non-space run is copied verbatim."""

    def __init__(self, punctuation):
        super().__init__("word", token_orders=TOKEN_ORDER)
        whitespace = pynini.union(" ", "\t", "\n", "\r", "\u00a0")
        # Exclude whole punctuation tokens, not punctuation embedded in words.
        self.graph = ((self.VCHAR - whitespace).plus - punctuation.graph).optimize()
        self.value_graph = (
            ((self.VCHAR - " ").plus - self.SPACE.plus)
            @ pynini.cdrewrite(pynini.cross("\u00a0", " "), "", "", self.VSIGMA)
            | pynutil.delete(self.SPACE.plus)
        ).optimize()
        self.tagger = self.add_tokens(self.tag_field("value", self.graph))
        self.verbalizer = self.delete_tokens(self.verbalize_field("value", self.value_graph))
