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

from pynini.lib.pynutil import delete

from tn.processor import Processor

TOKEN_ORDER = {}


class Word(Processor):
    """Words may contain inner punctuation (``कच्चा-पक्का``) but not start or end with it."""

    def __init__(self, punctuation):
        super().__init__("word", token_orders=TOKEN_ORDER)
        letters = (self.NOT_SPACE - punctuation.punct).plus
        self.graph = (letters + (punctuation.punct.plus + letters).star).optimize()
        self.tagger = self.add_tokens(self.tag_field("value", self.graph))
        self.verbalizer = self.delete_tokens(self.verbalize_field("value", self.graph | delete(self.SPACE.plus)))
