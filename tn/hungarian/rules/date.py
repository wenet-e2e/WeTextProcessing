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

from tn.hungarian.rules.utils import DIGIT, NONZERO, SIGMA, TO_LOWER, TO_UPPER, data_labels, data_map
from tn.processor import Processor

TOKEN_ORDER = {"date": ["year", "month", "day"]}

CASE_ENDINGS = {
    "e": "ét ének ével éért évé éig eként éül ében én énél ébe ére éhez éből éről étől",
    "a": "át ának ával áért ává áig aként ául ában án ánál ába ára ához ából áról ától",
}


def suffixed_days(days):
    """Day numbers with Hungarian case endings, e.g. ``1-jén`` -> ``elsején``."""

    output = []
    for number, day in days:
        for ending in ["je", "a", "e"]:
            if day.endswith(ending):
                output.append(("{}-{}".format(number, ending), day))
                break
        base = day[:-1]
        for ending in CASE_ENDINGS[day[-1]].split(" "):
            if day.endswith("eje"):
                output.append(("{}-j{}".format(number, ending), base + ending))
                output.append(("{}-{}".format(number, ending), base + ending))
            else:
                output.append(("{}-{}".format(number, ending), base + ending))
                output.append(("{}-{}".format(number, ending[1:]), base + ending))
        # Adjectival "-i" forms, e.g. "1-i" -> "elsei".
        output.append(("{}-i".format(number), base + "i"))
        if day == "elseje":
            output.append(("{}-ji".format(number), base + "i"))
        output.append(("{}-{}i".format(number, day[-1]), day + "i"))
    return output


class Date(Processor):
    """Dates, e.g. ``2010. ápr. 1-jén`` -> ``kettőezer-tíz április elsején``.

    Raw ``year``/``month``/``day`` fields; date dots and dashes are separators
    deleted by the tagger. Fields are verbalized in year-month-day order.
    """

    def __init__(self, cardinal):
        super().__init__("date", token_orders=TOKEN_ORDER)
        days = [(row[0], row[1]) for row in data_labels("dates/days.tsv")]
        optional_zero = delete("0").ques + NONZERO | NONZERO + DIGIT
        day_number = (optional_zero @ data_map("dates/days.tsv")).optimize()
        day_suffixed = pynini.string_map(suffixed_days(days))
        day_words = pynini.project(day_suffixed, "output") | pynini.project(day_number, "output")
        self.day = (day_number | day_suffixed | day_words).optimize()

        months = data_map("dates/months.tsv")
        month_number = (optional_zero @ months).optimize()
        names = pynini.project(months, "output")
        month_name = (names | (TO_LOWER + SIGMA) @ names).optimize()
        abbr = data_map("dates/month_abbr.tsv")
        roman = data_map("dates/months_roman.tsv")
        roman |= pynini.invert(pynini.invert(roman) @ pynini.closure(TO_UPPER))
        self.month = (month_number | month_name | abbr | roman).optimize()
        self.year = ((NONZERO + pynini.closure(DIGIT, 1, 3)) @ cardinal.plain).optimize()

        dot = delete(".")
        space = pynini.accep(" ")
        dash = pynini.cross("-", " ")
        year_t = self.tag_field("year", self.year)
        month_t = (
            self.tag_field("month", abbr | roman | month_number) + dot
            | self.tag_field("month", month_name)
        )
        day_t = self.tag_field("day", day_number) + dot.ques | self.tag_field("day", day_suffixed | day_words)
        month_number_t = self.tag_field("month", month_number)
        day_number_t = self.tag_field("day", day_number)
        graph = (year_t + dot + space).ques + month_t + space + day_t
        graph |= (year_t + dash).ques + month_number_t + dash + day_number_t
        graph |= year_t + dot + space + month_t
        graph |= day_number_t + dash + month_number_t + (dash + year_t).ques
        for separator in [".", "/"]:
            separator = pynini.cross(separator, " ")
            graph |= day_number_t + separator + month_number_t + separator + year_t
        self.tagger = self.add_tokens(graph)

        year_v = self.verbalize_field("year", self.year) + space
        month_v = self.verbalize_field("month", self.month)
        day_v = space + self.verbalize_field("day", self.day)
        self.verbalizer = self.delete_tokens(year_v.ques + month_v + day_v.ques)
