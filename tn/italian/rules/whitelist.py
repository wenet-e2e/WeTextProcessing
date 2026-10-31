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

from tn.italian.rules.utils import preferred_map
from tn.processor import Processor


class WhiteList(Processor):
    """Abbreviations, e.g. ``sig.na`` -> ``signorina``."""

    def __init__(self):
        super().__init__("whitelist", token_orders={})
        graph = preferred_map("italian/data/whitelist/whitelist.tsv")
        self.tagger = self.add_tokens(self.tag_field("value", graph))
        self.verbalizer = self.delete_tokens(self.verbalize_field("value", graph))
