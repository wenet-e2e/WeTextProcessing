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

import pynini
from pynini.lib.pynutil import delete

from tn.hindi.rules.utils import hindi_path
from tn.processor import Processor
from tn.utils import load_labels

TOKEN_ORDER = {}


class WhiteList(Processor):
    """``डॉ.`` -> ``whitelist { value: "डॉ." }`` -> ``डॉक्टर``."""

    def __init__(self):
        super().__init__("whitelist", token_orders=TOKEN_ORDER)
        graph = pynini.string_map(load_labels(hindi_path("whitelist/abbreviations.tsv")))
        # Dotted upper-case acronyms ("U.S.A." / "U. S. A") drop their dots.
        for dot in (".", ". "):
            graph |= self.UPPER + pynini.closure(delete(dot) + self.UPPER, 2) + delete(".").ques
        self.graph = graph.optimize()
        self.tagger = self.add_tokens(self.tag_field("value", self.graph))
        self.verbalizer = self.delete_tokens(self.verbalize_field("value", self.graph))
