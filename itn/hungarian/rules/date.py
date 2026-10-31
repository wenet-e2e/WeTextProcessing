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
from pynini.lib.pynutil import insert

from itn.hungarian.rules.utils import DIGIT, NONZERO, TO_LOWER, SIGMA, data_labels
from tn.processor import Processor

TOKEN_ORDER = {}

CASE_ENDINGS = {
    "e": "ét ének ével éért évé éig eként éül ében én énél ébe ére éhez éből éről étől",
    "a": "át ának ával áért ává áig aként ául ában án ánál ába ára ához ából áról ától",
}


def day_forms():
    """Spoken day -> written day, e.g. ``elseje`` -> ``1.``, ``elsején`` -> ``1-jén``."""

    written = {}
    for number, day in data_labels("dates/days.tsv"):
        written.setdefault(day, number + ".")
        base = day[:-1]
        for ending in CASE_ENDINGS[day[-1]].split(" "):
            prefix = "j" if day.endswith("eje") else ""
            written.setdefault(base + ending, "{}-{}{}".format(number, prefix, ending))
        written.setdefault(base + "i", "{}-{}i".format(number, "j" if day == "elseje" else ""))
    return pynini.string_map([(spoken, number) for spoken, number in written.items()]).optimize()


class Date(Processor):
    """Dates, e.g. ``kettőezer-tíz április elseje`` -> ``2010. április 1.``.

    Raw spoken ``year``/``month``/``day`` fields, written in Hungarian
    year-month-day order with the year dot.
    """

    def __init__(self, cardinal):
        super().__init__("date", ordertype="itn", token_orders=TOKEN_ORDER)
        names = pynini.union(*[row[1] for row in data_labels("dates/months.tsv")])
        self.month = (names | (TO_LOWER + SIGMA) @ names).optimize()
        self.day = day_forms()
        self.year = (pynini.invert(cardinal.spoken) @ (NONZERO + DIGIT ** 3) + insert(".")).optimize()

        space = pynini.accep(" ")
        year_t = self.tag_field("year", self.year) + space
        month_t = self.tag_field("month", self.month)
        day_t = space + self.tag_field("day", self.day)
        self.tagger = self.add_tokens(year_t.ques + month_t + day_t | year_t + month_t)

        year_v = self.verbalize_field("year", self.year) + space
        month_v = self.verbalize_field("month", self.month)
        day_v = space + self.verbalize_field("day", self.day)
        self.verbalizer = self.delete_tokens(year_v.ques + month_v + day_v.ques)
