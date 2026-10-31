# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from tn.processor import Processor


class Punctuation(Processor):
    """ASCII punctuation split off a neighbouring token, e.g. ``gatanu,``."""

    def __init__(self):
        super().__init__(name="punctuation", ordertype="itn", token_orders={})
        graph = pynini.union(*[pynini.escape(mark) for mark in "!#$%&'()*+,-./:;<=>?@^_`{|}~"]).optimize()
        self.tagger = self.add_tokens(self.tag_field("value", graph))
        self.verbalizer = self.delete_tokens(self.verbalize_field("value", graph))
