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
from pynini.lib.pynutil import add_weight, delete, insert

from tn.processor import Processor
from tn.utils import get_abs_path


TOKEN_ORDER = {"date": ["day", "month", "year"]}


class Date(Processor):
    """``le 02.03.2003`` -> ``date { day: "le 02" month: "03" year: "2003" }``.

    The optional determiner stays inside the day field, as in NeMo.
    """

    def __init__(self, cardinal):
        super().__init__("date", token_orders=TOKEN_ORDER)
        months = pynini.string_file(get_abs_path("french/data/dates/months.tsv"))
        eras = pynini.string_file(get_abs_path("french/data/dates/eras.tsv"))
        number = cardinal.graph

        leading_zero = (delete("0") | self.DIGIT - "0") + self.DIGIT | self.DIGIT
        determiner = (pynini.accep("le ") | pynini.accep("les ")).ques
        day_numbers = pynini.union(*[str(x) for x in range(1, 32)])
        # "1" reads as "premier"; the cardinal reading stays as an alternative.
        day_words = pynini.cross("1", "premier") | add_weight(number, 0.01)
        day = (determiner + (leading_zero @ day_numbers @ day_words)).optimize()
        month_number = leading_zero @ pynini.union(*[str(x) for x in range(1, 13)]) @ months
        month_name = months.copy().project("output").optimize()
        year = (((self.DIGIT - "0") + pynini.closure(self.DIGIT, 1, 3)) @ number).optimize()

        def field(name, graph):
            return insert(" ") + self.tag_field(name, graph)

        taggers = []
        for separator in ["/", ".", "-"]:
            sep = delete(separator)
            year_tagger = (sep + field("year", year)).ques
            taggers.append(self.tag_field("day", day) + sep + field("month", month_number) + year_tagger)
        named = delete(" ") + field("month", month_name) + (delete(" ") + field("year", year)).ques
        taggers.append(self.tag_field("day", day) + named)
        for separator in ["-", "/"]:
            days = (day + pynini.cross(separator, " ")).plus + day
            taggers.append(self.tag_field("day", days) + named)
        taggers.append(self.tag_field("year", eras))
        self.tagger = self.add_tokens(pynini.union(*taggers))

        days = (day + pynini.cross("-", " ")).plus + day | (day + pynini.cross("/", " ")).plus + day
        day_out = self.verbalize_field("day", day | days)
        month_out = pynini.accep(" ") + self.verbalize_field("month", month_number | month_name)
        year_out = pynini.accep(" ") + self.verbalize_field("year", year)
        self.verbalizer = self.delete_tokens(day_out + month_out + year_out.ques | self.verbalize_field("year", eras))
