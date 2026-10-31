# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pynini.lib.pynutil import delete, insert

from tn.processor import Processor
from tn.utils import get_abs_path, load_labels


# ISO dates are tagged year-first; every date is spoken day, month, year.
TOKEN_ORDER = {"date": ["day", "month", "year", "ordinal", "era"]}


class Date(Processor):
    """Numeric dates (``15/01/2024``, ``tháng 04/1969``, ``năm 200 TCN``) with raw day/month/year fields."""

    def __init__(self, cardinal):
        super().__init__("date", token_orders=TOKEN_ORDER)
        self.day_graph = pynini.string_map(load_labels(get_abs_path("vietnamese/data/date/days.tsv"))).optimize()
        self.month_graph = pynini.string_map(load_labels(get_abs_path("vietnamese/data/date/months.tsv"))).optimize()
        self.year_graph = (pynini.closure(self.DIGIT, 1, 4) @ cardinal.graph).optimize()
        year_raw = (pynini.closure(self.DIGIT, 1, 4) @ cardinal.raw).optimize()
        eras = {}
        for abbr, full in load_labels(get_abs_path("vietnamese/data/date/year_suffix.tsv")):
            eras[abbr.lower()] = full
            eras[abbr.upper()] = full
        self.era_graph = pynini.string_map(eras.items()).optimize()

        separator = delete(pynini.union("/", "-", ".")) + insert(" ")
        day = self.tag_field("day", self.day_graph)
        month = self.tag_field("month", self.month_graph)
        year = self.tag_field("year", year_raw)
        era = pynini.accep(" ") + self.tag_field("era", self.era_graph)
        day_prefix = delete("ngày ")
        month_word = delete(" tháng ") + insert(" ")
        year_word = delete(" năm ") + insert(" ")
        self.fields_tagger = pynini.union(
            day_prefix.ques + day + separator + month + separator + year,
            year + separator + month + separator + day,
            delete("tháng ") + month + (separator | pynini.accep(" ")) + year,
            day_prefix + day + (separator | month_word) + month,
            day_prefix + day + month_word + month + year_word + year,
            delete("năm ") + year + era.ques,
            delete("năm thứ ") + self.tag_field("ordinal", year_raw) + era,
        ).optimize()
        self.tagger = self.add_tokens(self.fields_tagger)

        def spoken(name, graph, prefix, leading_space):
            return insert(prefix) + self.verbalize_field(name, graph, leading_space)

        def day_v(lead):
            return spoken("day", self.day_graph, "ngày ", lead)

        def month_v(lead):
            return spoken("month", self.month_graph, "tháng ", lead)

        def year_v(lead):
            return spoken("year", self.year_graph, "năm ", lead)

        era_v = insert(" ") + self.verbalize_field("era", self.era_graph, True)
        space = insert(" ")
        self.fields_verbalizer = pynini.union(
            day_v(False) + space + month_v(True) + (space + year_v(True)).ques,
            month_v(False) + space + year_v(True),
            year_v(False) + era_v.ques,
            spoken("ordinal", self.year_graph, "năm thứ ", False) + era_v,
        ).optimize()
        self.verbalizer = self.delete_tokens(self.fields_verbalizer)
