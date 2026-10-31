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

from itn.spanish.rules.utils import CHAR, SIGMA, TO_LOWER, cased, data_file, int_to_roman
from tn.processor import Processor

TOKEN_ORDER = {}


def numbers_1_to_100():
    digit = data_file("numbers/digit.tsv")
    ties = data_file("numbers/ties.tsv")
    return pynini.union(
        digit,
        data_file("numbers/twenties.tsv"),
        data_file("numbers/teen.tsv"),
        ties + pynutil.insert("0"),
        ties + pynutil.delete(" y ") + digit,
    ).optimize()


class Date(Processor):
    """Raw spoken day and month ("primero de enero"), centuries ("siglo
    diecinueve") and years with an era ("doscientos tres antes de cristo").

    With cased input a capitalized day also capitalizes the month
    ("Uno de enero" -> "1 de Enero").
    """

    def __init__(self, cardinal, cased_input=False):
        super().__init__("date", ordertype="itn", token_orders=TOKEN_ORDER)
        day = numbers_1_to_100() @ pynini.union(*[str(i) for i in range(1, 32)])
        day = (day | pynini.cross("primero", "1")).optimize()
        self.day_graph = cased(day, cased_input)
        month = pynini.project(data_file("dates/months.tsv"), "input")
        suffix = pynini.invert(data_file("dates/year_suffix.tsv"))
        prefix = pynini.union("siglo ", "año ")
        if cased_input:
            month |= pynini.project(data_file("dates/months_cased.tsv"), "input")
            suffix |= pynini.invert(data_file("dates/year_suffix_cased.tsv"))
            prefix |= pynini.union("Siglo ", "Año ")
        self.month_graph = month.optimize()
        roman = prefix + int_to_roman(cardinal.graph)
        era = cardinal.graph + pynini.accep(" ") + suffix
        self.year_graph = (roman | era).optimize()

        of = cased(pynini.accep("de"), cased_input)
        day_month_t = (
            self.tag_field("day", self.day_graph) + pynutil.delete(" ") + pynutil.delete(of) + pynini.accep(" ")
            + self.tag_field("month", self.month_graph)
        )
        self.tagger = self.add_tokens(day_month_t | self.tag_field("year", self.year_graph))

        month_v = self.verbalize_field("month", self.month_graph, leading_space=True)
        day_month_v = self.verbalize_field("day", self.day_graph) + pynutil.insert(" de ") + month_v
        if cased_input:
            upper = pynini.project(TO_LOWER, "input")
            capitalized_day = (upper + SIGMA) @ self.day_graph
            lower_day = (pynini.difference(CHAR, upper) + SIGMA) @ self.day_graph
            capitalize = (pynini.invert(TO_LOWER) | upper) + SIGMA
            capitalized_month = self.verbalize_field(
                "month", (self.month_graph @ capitalize).optimize(), leading_space=True
            )
            day_month_v = (
                self.verbalize_field("day", lower_day.optimize()) + pynutil.insert(" de ") + month_v
                | self.verbalize_field("day", capitalized_day.optimize()) + pynutil.insert(" de ") + capitalized_month
            )
        self.verbalizer = self.delete_tokens(day_month_v | self.verbalize_field("year", self.year_graph))
