# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
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

from itn.utils import get_abs_path
from itn.vietnamese.rules.cardinal import DELETE_EXTRA_SPACE, DELETE_SPACE, numbers_file
from tn.processor import Processor


class Date(Processor):
    """``hai tư tháng bảy năm hai nghìn mười ba`` keeps day/month/year words and becomes ``24 tháng 7 năm 2013``."""

    def __init__(self, cardinal):
        super().__init__(name="date", ordertype="itn", token_orders={})
        ds = DELETE_SPACE
        sep = DELETE_EXTRA_SPACE + insert(" ")
        teen = numbers_file("teen")
        digit = numbers_file("digit")
        zero = numbers_file("zero")
        ties = numbers_file("ties")
        ten = delete("mươi")
        two_digits = pynini.union(
            ties + (ds + ten).ques + ds + (digit | cardinal.one | cardinal.four | cardinal.five),
            ties + ds + ten + insert("0", weight=0.01),
        )
        linh = pynini.cross(pynini.union("linh", "lẻ"), "0")
        zero_digits = linh + ds + (digit | cardinal.four) | zero + ds + digit
        tail = pynini.union(teen, two_digits, zero_digits)
        hundreds = (digit | zero) + ds + delete("trăm") | insert("0", weight=0.01)
        self.year_graph = pynini.union(
            digit + ds + (digit | zero) + ds + tail,
            digit + ds + delete(pynini.union("nghìn", "ngàn")) + ds + hundreds + ds + tail,
            digit + ds + delete("trăm") + ds + tail,
            digit + insert("0", weight=0.01) + ds + pynini.union(two_digits, zero_digits, teen),
            ((digit | zero) + ds) ** 3 + (digit | zero),
        ).optimize()
        self.month_graph = pynini.string_file(get_abs_path("vietnamese/data/months.tsv")).optimize()
        # A bare "tháng năm" stays text: "năm" is far more often "year" than "May".
        month_only = pynini.difference(self.input_projection(self.month_graph), "năm") @ self.month_graph
        self.day_graph = cardinal.graph_no_exception

        day = self.tag_field("day", self.day_graph)
        month = self.tag_field("month", self.month_graph)
        year = DELETE_EXTRA_SPACE + delete("năm") + sep + self.tag_field("year", self.year_graph)
        year = add_weight(year, -0.1)
        self.tagger = self.add_tokens(pynini.union(
            day + ds + delete("tháng") + sep + month + year.ques,
            delete("tháng") + ds + month + year,
            delete("tháng") + ds + self.tag_field("month", month_only),
            add_weight(delete("năm") + sep + self.tag_field("year", self.year_graph), -0.1),
        ))

        space = delete(" ")
        day_v = self.verbalize_field("day", self.day_graph)
        month_v = insert("tháng ") + self.verbalize_field("month", self.month_graph)
        year_v = insert("năm ") + self.verbalize_field("year", self.year_graph)
        self.verbalizer = self.delete_tokens(pynini.union(
            day_v + insert(" ") + space + month_v + (insert(" ") + space + year_v).ques,
            month_v + (insert(" ") + space + year_v).ques,
            year_v,
        ))
