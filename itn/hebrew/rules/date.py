# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

from itn.hebrew.rules.utils import (DELETE_SPACE, PREFIX, SPACE, VSIGMA, accep_file, first_field, next_field,
                                    string_file, trim)
from tn.processor import Processor


class Date(Processor):
    """Spoken dates, e.g. ``בשבעה עשר באוגוסט עשרים שלושים`` -> ``ב-17.8.2030``.

    ``day`` keeps a cardinal or ordinal day, ``month`` keeps the month word
    with its ``ב``/``ל`` prefix, ``year`` keeps the spoken year. A month name
    without a day stays a word (``בינואר 2021``), and ``year_prefix`` keeps
    ``שנת``/``השנה`` (``בשנת 2005``).
    """

    def __init__(self, cardinal):
        super().__init__(name="date", ordertype="itn", token_orders={})
        sep = delete(SPACE.plus) + insert(" ")
        ordinal = (VSIGMA + string_file("ordinals/digit.tsv")) @ cardinal.graph
        day = add_weight(cardinal.graph_two_digit | ordinal, -0.7)
        day = trim(day @ pynini.closure(pynini.union(*"0123456789"), 1, 2))
        delete_prefix = delete(PREFIX).ques
        month_name = string_file("months_name2number.tsv")
        month_any = (month_name | string_file("months_ordinal2number.tsv")).optimize()
        year = (cardinal.graph_two_digit + DELETE_SPACE + cardinal.graph_two_digit | cardinal.graph_thousands)
        year = trim(year)

        prefix = (self.tag_field("prefix", PREFIX) + insert(" ")).ques
        tag_day = prefix + self.tag_field("day", day) + sep
        tag_year = sep + self.tag_field("year", year)
        dm = tag_day + self.tag_field("month", delete_prefix + month_name)
        dmy = tag_day + self.tag_field("month", delete_prefix + month_any) + tag_year
        my = self.tag_field("month", PREFIX.ques + accep_file("months.tsv")) + tag_year
        year_prefix = PREFIX.ques + pynini.union("שנה", "שנת")
        y_only = self.tag_field("year_prefix", year_prefix) + tag_year
        self.tagger = self.add_tokens(dm | dmy | my | y_only)

        verbalize_day = (first_field(self, "prefix", PREFIX + insert("-")).ques
                         + self.verbalize_field("day", day + insert(".")))
        dmy = verbalize_day + next_field(self, "month", delete_prefix + month_any) + next_field(
            self, "year", insert(".") + year).ques
        my = self.verbalize_field("month", PREFIX.ques + accep_file("months.tsv")) + next_field(
            self, "year", insert(" ") + year)
        y_only = self.verbalize_field("year_prefix", year_prefix) + next_field(self, "year", insert(" ") + year)
        self.verbalizer = self.delete_tokens(dmy | my | y_only)
