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
from pynini.lib.pynutil import delete

from tn.hungarian.rules.utils import data_map, suffix_rewrite
from tn.processor import Processor

TOKEN_ORDER = {}


class Ordinal(Processor):
    """Ordinals written with a trailing dot, e.g. ``31.`` -> ``harmincegyedik``.

    The tagger keeps the digits and drops the ordinal dot; the verbalizer
    reads the cardinal and rewrites its last word into the ordinal form.
    """

    def __init__(self, cardinal):
        super().__init__("ordinal", token_orders=TOKEN_ORDER)
        to_ordinal = suffix_rewrite(data_map("ordinals/endings.tsv"), data_map("ordinals/exceptional.tsv"))
        self.bare = (cardinal.plain @ to_ordinal).optimize()
        self.graph = (cardinal.graph @ to_ordinal).optimize()
        # Denominators: "harmadik" -> "harmad", "második" -> "fél".
        exceptions = pynini.string_map([("első", "egyed"), ("második", "fél")])
        self.fractional = (self.bare @ suffix_rewrite(delete("ik"), exceptions)).optimize()
        self.tagger = self.add_tokens(self.tag_field("integer", self.graph) + delete("."))
        self.verbalizer = self.delete_tokens(self.verbalize_field("integer", self.graph))
