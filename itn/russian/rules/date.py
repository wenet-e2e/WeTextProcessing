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

from itn.russian.rules.utils import DIGIT, RU_ALPHA_OR_SPACE, data_path, number_names
from tn.processor import Processor

YEAR_SINGULAR = ["год", "года", "году", "годом", "годе"]
YEAR_PLURAL = ["годы", "годов", "годам", "годами", "годам", "годах"]


class Date(Processor):
    """Spoken dates, e.g. "первое февраля" -> date { day: "первое" month: "февраля" } -> "01.02"."""

    def __init__(self):
        super().__init__(name="date", ordertype="itn")
        ordinal = number_names().ordinal
        text = pynini.closure(RU_ALPHA_OR_SPACE)

        def ending(*endings):
            return (text + pynini.union(*endings)).optimize()

        def without(acceptor):
            return pynini.difference(text, acceptor).optimize()

        # Written -> spoken readings as in TN, inverted below.
        zero = pynutil.add_weight(pynini.cross("0", ""), -0.1) | pynutil.add_weight(pynini.cross("0", "ноль "), 0.1)
        zero_digit = zero + (DIGIT @ ordinal)
        digit_day = ((pynini.union("1", "2", "3") + DIGIT) | DIGIT) @ ordinal
        day = (zero_digit | digit_day) @ without(ending("ой", "ая", "ых", "ые", "ыми"))

        abbr_to_name = pynini.string_file(data_path("months/abbr_to_name.tsv"))
        number_to_abbr = pynini.string_file(data_path("months/numbers.tsv"))
        month_digits = (pynutil.add_weight(pynini.cross("0", ""), -0.1) | "1") + DIGIT | DIGIT
        month = month_digits @ number_to_abbr @ abbr_to_name
        month |= pynutil.add_weight(zero_digit | (pynini.accep("1") + DIGIT) @ ordinal, 0.1)

        year = ((DIGIT ** 4 | DIGIT ** 2) @ ordinal) | zero_digit
        genitive = ending("ой", "ого")
        year = (year @ without(genitive)) | pynutil.add_weight(year @ genitive, -0.1)
        wrong_start = pynini.union("две тысяча", "два тысяч", "два тысячи", "две тысяч ") + text
        year = year @ without(wrong_start) @ without(ending("ую", "ая"))
        year_word = pynini.cross("г.", pynini.union(*YEAR_SINGULAR)) | pynini.cross("гг.", pynini.union(*YEAR_PLURAL))
        year_word = (
            pynutil.add_weight(pynutil.insert(" "), -0.1) | pynutil.add_weight(pynini.accep(" "), 0.1)
        ) + year_word

        self.day = day.invert().optimize()
        self.month = month.invert().optimize()
        self.year = (year + year_word.ques).invert().optimize()
        self.year_only = (year + year_word).invert().optimize()

        space = pynutil.delete(" ") + pynutil.insert(" ")
        day_month = self.tag_field("day", self.day) + space + self.tag_field("month", self.month)
        self.tagger = self.add_tokens(
            day_month + (space + self.tag_field("year", self.year)).ques
            | self.tag_field("year", self.year_only)
        )
        dot = pynutil.delete(" ") + pynutil.insert(".")
        day_month = self.verbalize_field("day", self.day) + dot + self.verbalize_field("month", self.month)
        self.verbalizer = self.delete_tokens(
            day_month + (dot + self.verbalize_field("year", self.year)).ques
            | self.verbalize_field("year", self.year_only)
        )
