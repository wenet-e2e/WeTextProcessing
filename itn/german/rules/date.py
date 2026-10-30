# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pynini
from pynini.lib import pynutil

from tn.processor import Processor


TOKEN_ORDER = {"date": ["day", "month", "year"]}


class Date(Processor):

    def __init__(self, cardinal, tn_date, tn_ordinal):
        super().__init__("date", ordertype="itn", token_orders=TOKEN_ORDER)
        ordinal = pynini.cdrewrite(tn_ordinal.stem, "", "[EOS]", self.VSIGMA) + pynutil.insert("ter")
        # Source ITN inverts the verbalizer's ordinals, not TN's calendar bounds.
        self.day_graph = (ordinal.invert() @ cardinal.graph_no_exception).optimize()
        self.numeric_month_graph = (self.day_graph @ pynini.closure(self.DIGIT, 1, 2)).optimize()
        self.named_month_graph = tn_date.month_abbr_graph.invert().optimize()
        self.month_graph = (self.named_month_graph | self.numeric_month_graph).optimize()
        omit_spaces = pynini.closure(self.VSIGMA | pynutil.delete(" ", weight=0.0001))
        self.year_graph = (tn_date.year_graph @ omit_spaces).invert().optimize()

        pad = (self.DIGIT + self.DIGIT) | (pynutil.insert("0") + self.DIGIT)
        day = self.tag_field("day", self.day_graph)
        numeric_day = self.tag_field("day", self.day_graph @ pad)
        numeric_month = self.tag_field("month", self.numeric_month_graph)
        named_month = self.tag_field("month", self.named_month_graph)
        year = self.tag_field("year", self.year_graph)
        optional_year = pynini.closure(pynini.accep(" ") + year, 0, 1)
        self.fields_tagger = (
            day + pynini.accep(" ") + named_month + optional_year
            | numeric_day + pynini.accep(" ") + numeric_month + optional_year
            | named_month + pynini.accep(" ") + year
            | year
        ).optimize()
        self.tagger = self.add_tokens(self.fields_tagger)

        day_named = self.verbalize_field("day", self.day_graph) + pynutil.insert(".")
        day_numeric = self.verbalize_field("day", self.day_graph @ pad) + pynutil.insert(".")
        month_named = self.verbalize_field("month", self.named_month_graph)
        month_numeric = self.verbalize_field("month", self.numeric_month_graph @ pad) + pynutil.insert(".")
        year_v = self.verbalize_field("year", self.year_graph)
        named_date = (
            pynini.closure(day_named + pynini.accep(" "), 0, 1) + month_named
            + pynini.closure(pynini.accep(" ") + year_v, 0, 1)
        )
        numeric_date = (
            day_numeric + pynutil.delete(" ") + month_numeric
            + pynini.closure(pynutil.delete(" ") + year_v, 0, 1)
        )
        self.fields_verbalizer = (named_date | numeric_date | year_v).optimize()
        self.verbalizer = self.delete_tokens(self.fields_verbalizer)
