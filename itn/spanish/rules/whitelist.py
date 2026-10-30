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

from itn.spanish.rules.utils import data_labels
from tn.processor import Processor

TOKEN_ORDER = {}


class Whitelist(Processor):
    """Raw whitelisted phrases ("usted"); V writes the abbreviation ("Ud.")."""

    def __init__(self, cased_input=False):
        super().__init__("whitelist", ordertype="itn", token_orders=TOKEN_ORDER)
        labels = [(written, spoken) for written, spoken in data_labels("whitelist.tsv")]
        if cased_input:
            extra = []
            for written, spoken in labels:
                capitalized = written[0].upper() + written[1:]
                extra.append((capitalized, spoken.capitalize()))
                extra.append((capitalized, spoken.upper().replace(" Y ", " y ")))
                no_space = spoken.replace(" ", "")
                if len(spoken) == 2 * len(no_space) - 1:
                    extra.append((written, no_space))
                    extra.append((capitalized, no_space.upper()))
            labels += extra
        # Spoken form -> written form; the first written form of a phrase wins.
        mapping = {}
        for written, spoken in labels:
            mapping.setdefault(spoken, written)
        self.graph = pynini.string_map(sorted(mapping.items())).optimize()
        self.tagger = self.add_tokens(self.tag_field("value", self.graph))
        self.verbalizer = self.delete_tokens(self.verbalize_field("value", self.graph))
