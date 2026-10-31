# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from tn.processor import Processor
from tn.utils import get_abs_path, load_labels


TOKEN_ORDER = {"date": ["day", "month", "year"]}


class Date(Processor):
    """Numeric (DMY, ISO YMD, MDY) and textual dates, always read day-month-year."""

    def __init__(self, cardinal):
        super().__init__("date", token_orders=TOKEN_ORDER)
        months = load_labels(get_abs_path("portuguese/data/date/months.tsv"))
        phrases = {row[0]: row[1] for row in load_labels(get_abs_path("portuguese/data/date/verbal_phrases.tsv"))}
        preposition = phrases["preposition"]
        separators = [row[0] for row in load_labels(get_abs_path("portuguese/data/date/numeric_separators.tsv"))]

        days = [(str(n), n) for n in range(1, 32)] + [("{:02d}".format(n), n) for n in range(1, 10)]
        ordinal_units = {row[1]: row[0] for row in load_labels(get_abs_path("portuguese/data/ordinals/digit.tsv"))}
        first = ordinal_units[cardinal.digits["1"]]
        day_pairs = []
        for written, value in days:
            spoken = first if value == 1 else pynini.shortestpath(pynini.accep(str(value)) @ cardinal.graph).string()
            day_pairs.append((written, spoken))
        day = pynini.string_map(day_pairs).optimize()
        month_number = pynini.string_map(months).optimize()
        names = sorted({row[1] for row in months})
        month_name = pynini.string_map([(name, name) for name in names] + [(name.capitalize(), name) for name in names])
        month = (month_number | month_name).optimize()
        year = ((self.DIGIT - "0") + self.DIGIT**3) @ cardinal.graph

        def field(name, graph):
            return self.tag_field(name, graph)

        numeric = []
        for separator in separators:
            sep = delete(separator) + insert(" ")
            numeric.append(field("day", day) + sep + field("month", month_number) + sep + field("year", year))
            numeric.append(field("year", year) + sep + field("month", month_number) + sep + field("day", day))
            numeric.append(add_weight(
                field("month", month_number) + sep + field("day", day) + sep + field("year", year), 0.05))
        de = delete(self.SPACE.plus + preposition + self.SPACE.plus) + insert(" ")
        text = field("day", day) + de + field("month", month_name) + de + field("year", year)
        self.tagger = self.add_tokens(pynini.union(text, *numeric))

        glue = insert(" " + preposition + " ")
        self.verbalizer = self.delete_tokens(
            self.verbalize_field("day", day) + glue
            + self.verbalize_field("month", month, leading_space=True) + glue
            + self.verbalize_field("year", year, leading_space=True)
        )
