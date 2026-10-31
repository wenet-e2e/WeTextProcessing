# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from pynini.lib import pynutil

from itn.russian.rules.punctuation import WHITESPACE
from tn.processor import Processor


class Word(Processor):
    def __init__(self, punctuation):
        super().__init__(name="word", ordertype="itn")
        # Exclude whole punctuation tokens, not punctuation embedded in words.
        self.graph = ((self.VCHAR - WHITESPACE).plus - punctuation.graph).optimize()
        # Whitespace-only input is tagged as one word and verbalized to nothing.
        self.value_graph = (self.graph | pynutil.delete(self.SPACE.plus)).optimize()
        self.tagger = self.add_tokens(self.tag_field("value", self.graph))
        self.verbalizer = self.delete_tokens(self.verbalize_field("value", self.value_graph))
