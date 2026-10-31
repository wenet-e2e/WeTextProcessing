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
from pynini.lib import byte, pynutil

from itn.swedish.rules.utils import DIGIT, NONZERO, SPACE, table
from tn.processor import Processor

TOKEN_ORDER = {"date": ["year", "month", "day", "era"]}


def pad(graph):
    """Left-pads one- or two-digit outputs to two digits."""

    return (graph @ (pynutil.insert("0") + DIGIT | DIGIT**2)).optimize()


class Date(Processor):
    """Raw spoken dates (``andra januari tjugotjugotvå``) to ``2022-01-02``."""

    def __init__(self, cardinal, ordinal):
        super().__init__(name="date", ordertype="itn", token_orders=TOKEN_ORDER)
        day = (ordinal.graph @ pynini.union(*[str(x) for x in range(1, 32)])).optimize()
        months = table("dates/months")
        month_abbr = table("dates/month_abbr")
        era = table("dates/era_words")

        first = (cardinal.tens | cardinal.digit).optimize()
        second = (
            cardinal.tens @ (NONZERO + NONZERO)
            | pynutil.delete("hundra") + SPACE + pynutil.insert("0") + cardinal.digit
            | cardinal.tens @ (NONZERO + "0")
        )
        hundra = first + SPACE + pynutil.delete("hundra") + SPACE + second
        year = (
            first + SPACE + (second | pynini.cross("hundra", "00"))
            | first
            | cardinal.graph @ (NONZERO + pynini.closure(DIGIT, 1, 3))
            | hundra
        ).optimize()
        decade_second = cardinal.tens @ (NONZERO + "0") | pynini.cross("hundra", "00")
        decade_number = (
            first + SPACE + decade_second
            | decade_second
            | cardinal.graph @ (NONZERO + pynini.closure(DIGIT, 1, 2) + "0")
        )
        word = pynini.union("tal", "talet", "tals") | "tals" + (byte.ALPHA | pynini.union(*"åäöé")).plus
        decade = (decade_number + SPACE + pynutil.insert("-") + word).optimize()

        space = pynini.accep(" ")
        preserve = pynutil.insert(' preserve_order: "true"')
        year_t = self.tag_field("year", year)
        era_t = space + self.tag_field("era", era)
        day_month = self.tag_field("day", day) + space + self.tag_field("month", month_abbr) + preserve
        dmy = (
            self.tag_field("day", pad(day)) + space + self.tag_field("month", pad(months)) + space
            + year_t + era_t.ques
        )
        ymd = (
            year_t + space + self.tag_field("month", pad(months)) + space + self.tag_field("day", pad(day))
            + era_t.ques + preserve
        )
        self.tagger = self.add_tokens(year_t + era_t | day_month | dmy | ymd | self.tag_field("year", decade))

        era_v = pynutil.delete(" ") + pynutil.insert(" ") + self.verbalize_field("era", era)
        hyphen = pynutil.delete(" ") + pynutil.insert("-")
        full = (
            self.verbalize_field("year", year) + hyphen + self.verbalize_field("month", pad(months))
            + hyphen + self.verbalize_field("day", pad(day)) + era_v.ques
        )
        verbalizer = (
            full + pynutil.delete(' preserve_order: "true"').ques
            | self.verbalize_field("year", year) + era_v
            | self.verbalize_field("day", day) + pynutil.delete(" ") + pynutil.insert(" ")
            + self.verbalize_field("month", month_abbr) + pynutil.delete(' preserve_order: "true"')
            | self.verbalize_field("year", decade)
        )
        self.verbalizer = self.delete_tokens(verbalizer)
