# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from tn.swedish.rules.utils import alt, labels

TOKEN_ORDER = {}


class WhiteList(Processor):
    """Raw whitelist entries such as ``s:t``; nondeterministic mode adds units."""

    def __init__(self, deterministic=True):
        super().__init__("whitelist", token_orders=TOKEN_ORDER)
        self.deterministic = deterministic
        rows = labels("whitelist")
        graph = pynini.string_map(rows)
        if not deterministic:
            graph |= alt(pynini.string_map([[row[0].lower()] + row[1:] for row in rows]), 0.0001)
            for name in ("measure/unit", "measure/unit_neuter", "abbreviations_nondet"):
                graph |= pynini.string_map(labels(name))
        self.graph = graph.optimize()
        self.tagger = self.add_tokens(self.tag_field("value", self.graph))
        self.verbalizer = self.delete_tokens(self.verbalize_field("value", self.graph))
