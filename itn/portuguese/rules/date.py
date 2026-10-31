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
from pynini.lib.pynutil import delete, insert

from itn.utils import get_abs_path
from tn.processor import Processor


TOKEN_ORDER = {}


class Date(Processor):
    """Raw spoken dates: "primeiro de janeiro" -> 01 de janeiro, "dez do oito" -> 10/08."""

    def __init__(self, cardinal):
        super().__init__("date", ordertype="itn", token_orders=TOKEN_ORDER)
        one_to_hundred = pynini.union(
            insert("0") + cardinal.digit,
            cardinal.twenties,
            cardinal.teen,
            cardinal.ties + insert("0"),
            cardinal.ties + delete(" e ") + cardinal.digit,
        )
        days = pynini.union(*["{:02d}".format(day) for day in range(1, 32)])
        months = pynini.union(*["{:02d}".format(month) for month in range(1, 13)])
        day = (one_to_hundred @ days | pynini.cross("primeiro", "01")).optimize()
        month_number = (one_to_hundred @ months).optimize()
        month_name = pynini.string_file(get_abs_path("portuguese/data/months.tsv"))
        year = cardinal.digits_from_year
        space = self.DELETE_SPACE

        def joiner(word):
            return space + delete(word) + delete(self.SPACE.plus) + insert(" ")

        day_t = self.tag_field("day", day)
        year_t = joiner("de") + self.tag_field("year", year)
        self.tagger = self.add_tokens(
            day_t + joiner("de") + self.tag_field("month", month_name) + year_t.ques
            | day_t + joiner("do") + self.tag_field("month", month_number) + year_t.ques
        )

        day_v = self.verbalize_field("day", day)

        def tail(month, glue):
            return (
                insert(glue) + self.verbalize_field("month", month, leading_space=True)
                + (insert(glue) + self.verbalize_field("year", year, leading_space=True)).ques
            )

        self.verbalizer = self.delete_tokens(day_v + (tail(month_name, " de ") | tail(month_number, "/")))
