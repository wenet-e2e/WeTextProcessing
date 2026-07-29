# Copyright (c) 2022 Zhendong Peng (pzd17@tsinghua.org.cn)
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

from pynini import string_file
from pynini.lib.pynutil import delete, insert

from tn.processor import Processor
from tn.utils import get_abs_path


class Date(Processor):

    def __init__(self, range_tagger=None):
        super().__init__(name="date")
        self.range_tagger = range_tagger
        self.year = None
        self.month = None
        self.day = None
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        digit = string_file(get_abs_path("chinese/data/number/digit.tsv"))
        zero = string_file(get_abs_path("chinese/data/number/zero.tsv"))

        yyyy = digit + (digit | zero)**3
        m = string_file(get_abs_path("chinese/data/date/m.tsv"))
        mm = string_file(get_abs_path("chinese/data/date/mm.tsv"))
        d = string_file(get_abs_path("chinese/data/date/d.tsv"))
        dd = string_file(get_abs_path("chinese/data/date/dd.tsv"))
        rmsign = (delete("/") | delete("-") | delete(".")) + insert(" ")

        self.year = (yyyy + insert("年")).optimize()
        self.month = (m | mm).optimize()
        self.day = (d | dd).optimize()

        year = self.tag_field("year", self.year)
        month = self.tag_field("month", self.month)
        month_two_digit = self.tag_field("month", mm)
        day = self.tag_field("day", self.day)

        # yyyy/m/d | yyyy/mm/dd | dd/mm/yyyy
        # yyyy/0m | 0m/yyyy | 0m/dd
        date = ((year + rmsign + month + rmsign + day)
                | (day + rmsign + month + rmsign + year)
                | (year + rmsign + month_two_digit)
                | (month_two_digit + rmsign + year)
                | (month_two_digit + rmsign + day))
        tagger = self.add_tokens(date)

        if self.range_tagger is None:
            self.tagger = tagger
        else:
            self.tagger = tagger + (self.range_tagger + tagger).ques

    def build_verbalizer(self):
        year = delete('year: "') + self.year + delete('" ')
        month = delete('month: "') + self.month + delete('"')
        day = delete(' day: "') + self.day + delete('"')
        verbalizer = year.ques + month + day.ques
        self.verbalizer = self.delete_tokens(verbalizer)
