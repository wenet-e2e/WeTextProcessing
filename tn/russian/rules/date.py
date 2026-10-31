# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
from tn.russian.rules.utils import DIGIT, RU_ALPHA_OR_SPACE, number_names, ordered_map


TOKEN_ORDER = {}

YEAR_SINGULAR = ["год", "года", "году", "годом", "годе"]
YEAR_PLURAL = ["годы", "годов", "годам", "годами", "годам", "годах"]


class Date(Processor):
    """Day-month[-year] dates and "2009г." years with raw digit fields.

    "06.03.2006" -> date { day: "06" month: "03" year: "2006" }
    """

    def __init__(self):
        super().__init__("date", token_orders=TOKEN_ORDER)
        ordinal = number_names().ordinal
        text = pynini.closure(RU_ALPHA_OR_SPACE)

        def ending(*endings):
            return (text + pynini.union(*endings)).optimize()

        def without(acceptor):
            return pynini.difference(text, acceptor).optimize()

        def prefer(acceptor):
            # Ranks the usual date reading first; every NeMo reading stays available.
            return (pynutil.add_weight(acceptor, -0.5) | without(acceptor)).optimize()

        zero = pynutil.add_weight(pynini.cross("0", ""), -0.1) | pynutil.add_weight(pynini.cross("0", "ноль "), 0.1)
        zero_digit = zero + (DIGIT @ ordinal)
        valid_days = pynini.union(*[str(day) for day in range(1, 32)], *[f"0{day}" for day in range(1, 10)])
        digit_day = ((pynini.union("1", "2", "3") + DIGIT) | DIGIT) @ ordinal
        day = valid_days @ (zero_digit | digit_day) @ without(ending("ой", "ая", "ых", "ые", "ыми"))
        self.day = (day @ prefer(ending("ое", "ье"))).optimize()

        abbr_to_name = ordered_map("months/abbr_to_name.tsv")
        number_to_abbr = ordered_map("months/numbers.tsv")
        valid_months = pynini.union(*[str(month) for month in range(1, 13)], *[f"0{month}" for month in range(1, 10)])
        month_digits = (pynutil.add_weight(pynini.cross("0", ""), -0.1) | "1") + DIGIT | DIGIT
        month_name = (month_digits @ number_to_abbr @ abbr_to_name) | pynutil.add_weight(abbr_to_name, 0.1)
        digit_month = valid_months @ (zero_digit | (pynini.accep("1") + DIGIT) @ ordinal)
        self.month = (month_name | pynutil.add_weight(digit_month, 0.5)).optimize()

        year = ((DIGIT ** 4 | DIGIT ** 2) @ ordinal) | zero_digit
        genitive = ending("ой", "ого")
        year = (year @ without(genitive)) | pynutil.add_weight(year @ genitive, -0.1)
        wrong_start = pynini.union("две тысяча", "два тысяч", "два тысячи", "две тысяч ") + text
        year = year @ without(wrong_start) @ without(ending("ую", "ая")) @ prefer(ending("ого"))
        year_word = pynini.cross("г.", pynini.union(*YEAR_SINGULAR) | pynutil.add_weight("года", -0.1))
        year_word |= pynini.cross("гг.", pynini.union(*YEAR_PLURAL))
        year_word = (
            pynutil.add_weight(pynutil.insert(" "), -0.1) | pynutil.add_weight(pynini.accep(" "), 0.1)
        ) + year_word
        self.year = (year + year_word.ques).optimize()
        self.year_only = (year + year_word).optimize()

        separator = (
            pynutil.add_weight(pynutil.delete("."), 1.09)
            | pynutil.add_weight(pynutil.delete(pynini.union("/", "-")), 1.1)
        ) + pynutil.insert(" ")
        day_month = self.tag_field("day", self.day) + separator + self.tag_field("month", self.month)
        self.tagger = self.add_tokens(
            day_month + (separator + self.tag_field("year", self.year)).ques
            | self.tag_field("year", self.year_only)
        )
        day_month = (
            self.verbalize_field("day", self.day) + pynini.accep(" ") + self.verbalize_field("month", self.month)
        )
        self.verbalizer = self.delete_tokens(
            day_month + (pynini.accep(" ") + self.verbalize_field("year", self.year)).ques
            | self.verbalize_field("year", self.year_only)
        )
