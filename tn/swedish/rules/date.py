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
from tn.swedish.rules.utils import SV_ALPHA, alt, tsv

TOKEN_ORDER = {"date": ["day", "month", "year", "era"]}


class Date(Processor):
    """Raw day, month, year and era fields; spoken as day month year era."""

    def __init__(self, cardinal, ordinal, deterministic=True):
        super().__init__("date", token_orders=TOKEN_ORDER)
        self.deterministic = deterministic
        digit = self.DIGIT
        nonzero = digit - "0"
        months = tsv("dates/months")
        month_abbr = tsv("dates/month_abbr")
        era_words = tsv("dates/era_words")
        era = (tsv("dates/era_suffix") @ era_words | era_words.copy().project("output")).optimize()
        month_names = months.copy().project("output")
        dot = pynutil.delete(".").ques

        day_number = pynini.union(*[str(x) for x in range(1, 32)]) @ ordinal.bare_ordinals
        day_digits = (
            pynutil.delete("0") + (digit @ day_number)
            | (nonzero + digit) @ day_number
            | digit @ day_number
        ).optimize()
        day_words = day_digits.copy().project("output")
        day_numeric = day_digits
        day_written = (day_digits + dot | ordinal.suffixed_to_words | day_words).optimize()

        month_number = (
            ((pynutil.delete("0") | nonzero) + digit | digit) @ pynini.union(*[str(x) for x in range(1, 13)])
            @ months
        ).optimize()
        month_written = (month_names | month_abbr + dot).optimize()

        number = cardinal.graph
        first = ((nonzero + digit.ques) @ number).optimize()
        second = (
            (nonzero + nonzero) @ number
            | pynini.cross("0", "hundra") + (nonzero @ number)
            | (nonzero + "0") @ number
        )
        hundra = first + pynutil.insert("hundra") + second
        second |= pynini.cross("00", "hundra")
        year = first + second | first
        if not deterministic:
            year |= alt(((nonzero + pynini.closure(digit, 1, 3)) @ number))
            year |= alt(hundra)
            for joint in (" hundra", " hundra ", "hundra "):
                year |= alt(first + pynutil.insert(joint) + second)
            year |= alt(first + pynutil.insert(" ") + second)
        self.year = year.optimize()

        decade_second = (nonzero + "0") @ number | pynini.cross("00", "hundra")
        decade_number = first + decade_second | decade_second
        hyphen = pynutil.delete("-")
        if not deterministic:
            hyphen |= alt(pynini.cross("-", " "))
            decade_number |= alt((nonzero + pynini.closure(digit, 1, 2) + "0") @ number)
        decade_word = pynini.union("tal", "talet", "tals") | "tals" + SV_ALPHA.plus
        decade = (decade_number + hyphen + decade_word).optimize()

        def field(name, graph):
            return self.tag_field(name, graph), self.verbalize_field(name, graph, leading_space=True)

        space_t = pynutil.delete(pynini.accep(" ").star) + pynutil.insert(" ")
        ins = pynutil.insert(" ")
        comma = pynutil.delete(",").ques
        year_t, year_v = field("year", self.year)
        era_t, era_v = field("era", era)
        era_opt_t = (space_t + era_t).ques
        era_opt_v = (ins + era_v).ques

        day_t = self.tag_field("day", day_written)
        num_day_t = self.tag_field("day", day_numeric)
        month_t = self.tag_field("month", month_written)
        num_month_t = self.tag_field("month", month_number)
        year_first_t = self.tag_field("year", self.year)

        dmy = day_t + space_t + month_t + comma + (space_t + year_t + era_opt_t).ques
        my = month_t + comma + space_t + year_t + era_opt_t
        ymd = year_first_t + pynini.cross("-", " ") + num_month_t + (
            pynini.cross("-", " ") + self.tag_field("day", day_digits + dot)).ques
        for sep in (".", "-", "/"):
            sep_t = pynini.cross(sep, " ")
            dmy |= num_day_t + sep_t + num_month_t + (sep_t + year_t + era_opt_t).ques
            ymd |= year_first_t + sep_t + num_month_t + (sep_t + num_day_t).ques
        year_era = year_first_t + space_t + era_t
        decade_t = self.tag_field("year", decade)
        self.tagger = self.add_tokens(dmy | my | ymd | year_era | decade_t)

        day_v = self.verbalize_field("day", day_written)
        month_v = self.verbalize_field("month", month_written | month_number)
        month_lv = self.verbalize_field("month", month_written | month_number, leading_space=True)
        year_lv = year_v | self.verbalize_field("year", decade, leading_space=True)
        tail = (ins + year_lv + era_opt_v).ques
        verbalizer = (
            (day_v + ins + month_lv | month_v) + tail
            | self.verbalize_field("year", self.year) + ins + era_v
            | self.verbalize_field("year", decade)
        )
        self.verbalizer = self.delete_tokens(verbalizer)
