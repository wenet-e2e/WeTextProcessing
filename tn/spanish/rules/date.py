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
from pynini.lib import pynutil

from tn.processor import Processor
from tn.spanish.rules.utils import (
    DIGIT,
    data_file,
    inputs,
    rewrite,
    roman_to_int,
    strip_cardinal_apocope,
)

TOKEN_ORDER = {"date": ["weekday", "day", "month", "year"]}

ARTICLES = pynini.union("de", "del", "el", "del año", "año")


def _numbers(last):
    """Written numbers 1..last with an optional leading zero, unpadded on output."""
    plain = pynini.union(*[str(number) for number in range(1, last + 1)])
    padded = pynini.union(*[pynini.cross("0" + str(number), str(number)) for number in range(1, 10)])
    return (plain | padded).optimize()


class Date(Processor):
    """Raw day, month, year and weekday fields; V spells them out in d/m/y order."""

    def __init__(self, cardinal, deterministic=True):
        super().__init__("date", token_orders=TOKEN_ORDER)
        self.deterministic = deterministic
        months = data_file("dates/months.tsv")
        month_abbr = data_file("dates/months_abbr.tsv")
        weekdays = data_file("dates/days_abbr.tsv")
        year_suffix = data_file("dates/year_suffix.tsv")
        days = _numbers(31)
        month_numbers = (_numbers(12) @ months).optimize()
        month_names = pynini.project(months, "output") | month_abbr
        year = ((DIGIT - "0") + pynini.closure(DIGIT, 1, 3)) @ cardinal.graph
        year = strip_cardinal_apocope(year)
        roman = pynini.union("siglo ", "año ") + roman_to_int(cardinal.graph)
        era = (cardinal.graph | roman) + (pynini.accep(" ") | pynutil.insert(" ")) + year_suffix

        # Tagger: separators and the "de" after the day are structure only.
        space = pynini.accep(" ")
        weekday_t = self.tag_field("weekday", weekdays) + space
        day_t = self.tag_field("day", days)
        month_name_t = self.tag_field("month", month_names)
        month_number_t = self.tag_field("month", month_numbers)
        year_t = self.tag_field("year", year)
        year_article_t = self.tag_field("year", pynini.closure(ARTICLES + " ", 0, 1) + year)
        optional_year = pynini.closure(space + year_article_t, 0, 1)
        graph = (
            pynini.closure(weekday_t, 0, 1) + day_t + pynini.closure(pynutil.delete(" de"), 0, 1)
            + space + month_name_t + optional_year
        )
        graph |= month_name_t + pynini.cross(pynini.closure(" ", 1), " ") + day_t + optional_year
        for separator in (".", "-", "/"):
            separator = pynini.cross(separator, " ")
            optional_year = pynini.closure(separator + year_t, 0, 1)
            graph |= day_t + separator + month_number_t + optional_year
            if not deterministic:
                graph |= pynutil.add_weight(month_number_t + separator + day_t + optional_year, 0.01)
        graph |= (DIGIT ** 4 @ year_t) + pynini.cross("-", " ") + month_number_t + pynini.cross("-", " ") + day_t
        graph |= self.tag_field("year", roman | era)
        self.tagger = self.add_tokens(graph)

        # Verbalizer: "primero de mayo de mil novecientos noventa".
        day = strip_cardinal_apocope(days @ cardinal.graph)
        first = rewrite(day, pynini.cross("uno", "primero"), "[BOS]", "[EOS]")
        day = first if deterministic else (first | pynutil.add_weight(day, 0.001))
        day_v = self.verbalize_field("day", day)
        weekday_v = self.verbalize_field("weekday", weekdays) + space
        month_v = pynutil.insert(" de ") + self.verbalize_field(
            "month", inputs(pynini.project(months, "output")) | month_abbr | month_numbers, leading_space=True,
        )
        year_graph = ARTICLES + " " + year | pynutil.insert("de ") + year
        year_v = pynutil.insert(" ") + self.verbalize_field("year", year_graph, leading_space=True)
        graph = pynini.closure(weekday_v, 0, 1) + day_v + month_v + pynini.closure(year_v, 0, 1)
        graph |= self.verbalize_field("year", roman | era)
        self.verbalizer = self.delete_tokens(graph)
