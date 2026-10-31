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

from tn.hungarian.rules.utils import data_map, inflected_map
from tn.processor import Processor

TOKEN_ORDER = {}


class WhiteList(Processor):
    """Abbreviations with their case endings, e.g. ``tv.-nyel`` -> ``törvénnyel``."""

    def __init__(self):
        super().__init__("whitelist", token_orders=TOKEN_ORDER)
        graph = data_map("whitelist.tsv")
        graph |= inflected_map("whitelist_inflect.tsv")
        graph |= inflected_map("whitelist_inflect_sg.tsv", singular_only=True)
        self.graph = graph.optimize()
        self.tagger = self.add_tokens(self.tag_field("value", self.graph))
        self.verbalizer = self.delete_tokens(self.verbalize_field("value", self.graph))
