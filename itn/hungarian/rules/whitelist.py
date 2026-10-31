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

from itn.hungarian.rules.utils import data_labels, naive_inflector
from tn.processor import Processor

TOKEN_ORDER = {}

# Abbreviations whose expansions other rules write (quantities, units).
SKIPPED = {"Mrd", "mp"}


def abbreviations():
    """Spoken expansion (with case endings) -> abbreviation; the first listed abbreviation wins."""

    written = {}
    tables = [("whitelist.tsv", None), ("whitelist_inflect.tsv", False), ("whitelist_inflect_sg.tsv", True)]
    for name, singular_only in tables:
        for row in data_labels(name):
            abbr, expansion = row[0], row[1]
            if abbr in SKIPPED:
                continue
            written.setdefault(expansion, abbr)
            if singular_only is not None and " " not in expansion:
                for form, spoken in naive_inflector(abbr, expansion, singular_only):
                    written.setdefault(spoken, form)
    return pynini.string_map(list(written.items())).optimize()


class WhiteList(Processor):
    """Abbreviations with their case endings, e.g. ``törvénnyel`` -> ``tv.-nyel``."""

    def __init__(self):
        super().__init__("whitelist", ordertype="itn", token_orders=TOKEN_ORDER)
        self.graph = abbreviations()
        self.tagger = self.add_tokens(self.tag_field("value", self.graph))
        self.verbalizer = self.delete_tokens(self.verbalize_field("value", self.graph))
