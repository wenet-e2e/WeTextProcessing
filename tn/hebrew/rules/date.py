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

from tn.hebrew.rules import numbers
from tn.hebrew.rules.utils import (DEFINITE_PREFIX, DIGIT, INDEFINITE_PREFIX, PREFIX, SPACE, first_field, next_field)
from tn.processor import Processor

MONTHS = ["ינואר", "פברואר", "מרץ", "אפריל", "מאי", "יוני", "יולי", "אוגוסט", "ספטמבר", "אוקטובר", "נובמבר", "דצמבר"]


def _numbers(limit):
    return {number: {str(number), "%02d" % number} for number in range(1, limit + 1)}


def _relabel(readings):
    """Maps ``1``/``01`` style digits through a per-number reading."""

    pairs = [(form, reading(number)) for reading, limit in readings for number, forms in _numbers(limit).items()
             for form in forms]
    return pynini.string_map(pairs).optimize()


class Date(Processor):
    """Written dates, e.g. ``ב-17.8.2030`` -> ``בשבעה עשר באוגוסט עשרים שלושים``.

    Days are masculine cardinals; after a definite prefix (``ה-10.4``) days
    1-10 are ordinals (``העשירי באפריל``). A day and month without a year
    need a prefix, since ``18.6`` is a decimal. Years 2020-2099 are read in
    pairs (``עשרים עשרים ואחת``), other years as cardinals.
    """

    def __init__(self, cardinal):
        super().__init__(name="date")
        sep = delete(SPACE.plus) + insert(" ")
        day = pynini.project(_relabel([(str, 31)]), "input")
        month = pynini.project(_relabel([(str, 12)]), "input")
        pair = pynini.string_map([("20%02d" % year, "עשרים " + numbers.group(year)) for year in range(20, 100)])
        four = (DIGIT - "0") + DIGIT**3
        two = pynini.string_map([("%02d" % year, numbers.cardinal(year) if year >= 10 else "אפס " +
                                  numbers.FEMININE[year]) for year in range(100)])
        short = (DIGIT - "0") + pynini.closure(DIGIT, 0, 2)
        year = (pair | (pynini.project(four, "input") - pynini.project(pair, "input")) @ cardinal.number | two
                | (pynini.project(short, "input") - pynini.project(two, "input")) @ cardinal.number)
        year = year.optimize()
        long_year = (four @ year).optimize()
        name = pynini.union(*MONTHS)
        prefixed_name = (PREFIX.ques + name).optimize()
        year_prefix = (PREFIX.ques + pynini.union("שנה", "שנת")).optimize()

        prefix = self.tag_field("prefix", PREFIX) + delete("-") + insert(" ")
        numeric = (self.tag_field("day", day) + delete(".") + insert(" ") + self.tag_field("month", month))
        tag_year = self.tag_field("year", year)
        dmy = prefix.ques + numeric + delete(".") + insert(" ") + tag_year
        dm = prefix + numeric
        my = self.tag_field("month", prefixed_name) + sep + self.tag_field("year", long_year)
        y_only = self.tag_field("year_prefix", year_prefix) + sep + tag_year
        self.tagger = self.add_tokens(dmy | dm | my | y_only)

        masculine = _relabel([(lambda number: numbers.group(number, masculine=True), 31)])
        definite = _relabel([(lambda number: numbers.ORDINALS[number], 10)]) | (
            pynini.project(masculine, "input") - pynini.project(_relabel([(str, 10)]), "input")) @ masculine
        month_name = pynini.string_map([(form, " ב" + MONTHS[number - 1]) for number, forms in _numbers(12).items()
                                        for form in forms])
        day = (first_field(self, "prefix", DEFINITE_PREFIX) + self.verbalize_field("day", definite)
               | first_field(self, "prefix", INDEFINITE_PREFIX).ques + self.verbalize_field("day", masculine))
        year = next_field(self, "year", insert(" ") + year)
        dmy = day + next_field(self, "month", month_name) + year.ques
        my = self.verbalize_field("month", prefixed_name) + year
        y_only = self.verbalize_field("year_prefix", year_prefix) + year
        self.verbalizer = self.delete_tokens(dmy | my | y_only)
