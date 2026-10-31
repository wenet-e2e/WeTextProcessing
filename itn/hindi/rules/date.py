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
from pynini.lib.pynutil import add_weight, delete, insert

from itn.hindi.rules.utils import (DELETE_EXTRA_SPACE, DELETE_SPACE, HI_DIGIT, devanagari_numbers, spoken_to_written,
                                   string_file)
from tn.processor import Processor

TOKEN_ORDER = {}


class Date(Processor):
    """``पच्चीस मार्च दो हज़ार दस`` -> ``date { value: "पच्चीस मार्च दो हज़ार दस" }`` -> ``२५ मार्च, २०१०``."""

    def __init__(self, cardinal):
        super().__init__("date", ordertype="itn", token_orders=TOKEN_ORDER)
        number = cardinal.graph_no_exception
        year = add_weight(number @ pynini.closure(HI_DIGIT, 1, 4), 0.03)
        month = string_file("date/months.tsv")
        day = (cardinal.digit | cardinal.teens_and_ties) @ devanagari_numbers(1, 32)
        era = spoken_to_written("date/century.tsv")
        suffix = string_file("ordinals/morph_features.tsv")

        def join(*graphs):
            """Spoken parts separated by optional whitespace; written parts by a single space."""
            graph = graphs[0]
            for part in graphs[1:]:
                graph = graph + DELETE_SPACE + insert(" ") + part
            return graph

        comma = insert(",")
        year_range = year + DELETE_SPACE + pynini.cross("से", "-") + DELETE_SPACE + year
        graph = pynini.union(
            join(day, month),
            join(month, day),
            join(day, month + comma, year),
            join(month, day + comma, year),
            join(month, year),
            year,
            join(year, era),
            join(day, month + comma, year, era),
            join(month + comma, year, era),
            year_range,
            join(year_range, era),
            number + suffix + DELETE_EXTRA_SPACE + insert(" ") + era,
            month + DELETE_SPACE + delete("की") + DELETE_SPACE + insert(" ") + day,
        )
        self.graph = graph.optimize()
        self.tagger = self.add_tokens(self.tag_field("value", self.graph))
        self.verbalizer = self.delete_tokens(self.verbalize_field("value", self.graph))
