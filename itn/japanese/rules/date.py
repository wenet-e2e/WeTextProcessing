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
from tn.utils import get_abs_path


class Date(Processor):

    def __init__(self):
        super().__init__(name="date")
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        cardinal = Cardinal().ten_thousand_minus
        day = string_file(get_abs_path('../itn/japanese/data/date/day.tsv'))
        month = string_file(
            get_abs_path('../itn/japanese/data/date/month.tsv'))
        to = cross('から', '〜')

        # 一月 一日 一年
        year = (insert('year: "') + cardinal + (to + cardinal).ques +
                delete('年') + insert('"'))
        month = (insert('month: "') + month + (to + month).ques + delete('月') +
                 insert('"'))
        day = (insert('day: "') + day + (to + day).ques + delete('日') +
               insert('"'))

        # 二千二十四年十月一日 二千二十四年十月 十月一日
        graph_date = (year + insert(" ") + month
                      | month + insert(" ") + day
                      | year + insert(" ") + month + insert(" ") + day)

        # specific context for era year, e.g., L6 -> "令和6年"
        context = union(accep('今年は'), accep('来年は'), accep('再来年は'),
                        accep('去年は'), accep('一昨年は'), accep('おととしは'))
        era_year = union(cross("R", "令和"), cross("H", "平成"), cross("S", "昭和"),
                         cross("T", "大正"), cross("M", "明治"))
        era_year = context + era_year + cardinal
        era_year = insert('year: "') + era_year + insert('"')

        date = graph_date | era_year
        self.tagger = self.add_tokens(date).optimize()

    def build_verbalizer(self):
        year = delete('year: "') + self.SIGMA + insert('年') + delete('"')
        era_year = delete('year: "') + self.SIGMA + delete('"')
        month = delete('month: "') + self.SIGMA + insert('月') + delete('"')
        day = delete('day: "') + self.SIGMA + insert('日') + delete('"')

        graph_regular = (year + delete(' ') + month
                         | month + delete(' ') + day
                         | year + delete(' ') + month + delete(' ') + day)

        graph = graph_regular | era_year
        self.verbalizer = self.delete_tokens(graph).optimize()
