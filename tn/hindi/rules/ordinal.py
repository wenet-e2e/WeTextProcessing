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
from pynini.lib.pynutil import add_weight

from tn.hindi.rules.utils import hindi_path
from tn.processor import Processor

TOKEN_ORDER = {}


class Ordinal(Processor):
    """``१०वां`` -> ``ordinal { integer: "१०वां" }`` -> ``दसवां``."""

    def __init__(self, cardinal):
        super().__init__("ordinal", token_orders=TOKEN_ORDER)
        suffixes = pynini.union(
            pynini.string_file(hindi_path("ordinal/suffixes.tsv")),
            pynini.string_file(hindi_path("ordinal/suffixes_map.tsv")),
        )
        exceptions = add_weight(pynini.string_file(hindi_path("ordinal/exceptions.tsv")), -0.1)
        self.graph = pynini.union(exceptions, cardinal.limited_graph + suffixes).optimize()
        self.tagger = self.add_tokens(self.tag_field("integer", self.graph))
        self.verbalizer = self.delete_tokens(self.verbalize_field("integer", self.graph))
