# Copyright (c) 2024 Logan Liu (2319277867@qq.com)
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

from pynini import accep, cross, string_file, union
from pynini.lib.pynutil import delete, insert

from itn.japanese.rules.cardinal import Cardinal
from tn.processor import Processor
from itn.utils import get_abs_path


class Date(Processor):

    def __init__(self, cardinal=None, input_processor=None):
        super().__init__(name="date")
        self.cardinal = cardinal or Cardinal()
        self.input_processor = input_processor
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        cardinal = self.cardinal.ten_thousand_minus
        day = string_file(get_abs_path("japanese/data/date/day.tsv"))
        month = string_file(get_abs_path("japanese/data/date/month.tsv"))
        to = cross("から", "〜")

        # 一月 一日 一年
        self.year = self.apply_input_processor(cardinal + (to + cardinal).ques + delete("年"), self.input_processor)
        self.month = self.apply_input_processor(month + (to + month).ques + delete("月"), self.input_processor)
        self.day = self.apply_input_processor(day + (to + day).ques + delete("日"), self.input_processor)
        year = self.tag_field("year", self.year)
        month = self.tag_field("month", self.month)
        day = self.tag_field("day", self.day)

        # 二千二十四年十月一日 二千二十四年十月 十月一日
        graph_date = (year + insert(" ") + month | month + insert(" ") + day | year + insert(" ") + month + insert(" ") + day)

        # specific context for era year, e.g., L6 -> "令和6年"
        context = union(accep("今年は"), accep("来年は"), accep("再来年は"), accep("去年は"), accep("一昨年は"), accep("おととしは"))
        era_year = union(cross("R", "令和"), cross("H", "平成"), cross("S", "昭和"), cross("T", "大正"), cross("M", "明治"))
        self.era_year = self.apply_input_processor(context + era_year + cardinal, self.input_processor)
        era_year = self.tag_field("year", self.era_year)

        date = graph_date | era_year
        self.tagger = self.add_tokens(date).optimize()

    def build_verbalizer(self):
        year = self.verbalize_field("year", self.year) + insert("年")
        era_year = self.verbalize_field("year", self.era_year)
        month = self.verbalize_field("month", self.month) + insert("月")
        day = self.verbalize_field("day", self.day) + insert("日")

        graph_regular = (year + delete(" ") + month | month + delete(" ") + day
                         | year + delete(" ") + month + delete(" ") + day)

        graph = graph_regular | era_year
        self.verbalizer = self.delete_tokens(graph).optimize()
