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

import pynini
from pynini.lib import pynutil

from tn.processor import Processor
from tn.russian.rules.utils import RU_ALPHA, latin_to_cyrillic, ordered_rows


TOKEN_ORDER = {}


class WhiteList(Processor):
    """Abbreviations, standalone units ("м²", "%") and spelled Latin words ("abc")."""

    def __init__(self):
        super().__init__("whitelist", token_orders=TOKEN_ORDER)

        def lowered(name):
            return pynini.string_map([(row[0].lower(),) + row[1:] for row in ordered_rows(name)])

        # Single-letter units such as "м" stay words; "°" and "%" are read.
        units = (self.VCHAR ** (2, ...) | pynini.difference(self.VCHAR, RU_ALPHA)) @ lowered("measurements.tsv")
        to_cyrillic = latin_to_cyrillic()
        latin = to_cyrillic + pynini.closure(pynutil.insert(" ") + to_cyrillic)
        self.graph = (lowered("whitelist.tsv") | units | latin).optimize()
        self.tagger = self.add_tokens(self.tag_field("value", self.graph))
        self.verbalizer = self.delete_tokens(self.verbalize_field("value", self.graph))
