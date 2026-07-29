# Copyright (c) 2022 Xingchen Song (sxc19@tsinghua.org.cn)
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

from pynini import accep, string_file
from pynini.lib.pynutil import delete, insert

from tn.processor import Processor
from itn.utils import get_abs_path


class Date(Processor):

    def __init__(self):
        super().__init__(name="date")
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        digit = string_file(get_abs_path("chinese/data/number/digit.tsv"))  # 1 ~ 9
        zero = string_file(get_abs_path("chinese/data/number/zero.tsv"))  # 0

        yyyy = digit + (digit | zero)**3  # 二零零八年
        yyy = digit + (digit | zero)**2  # 公元一六八年
        yy = (digit | zero)**2  # 零八年奥运会
        mm = string_file(get_abs_path("chinese/data/date/mm.tsv"))
        dd = string_file(get_abs_path("chinese/data/date/dd.tsv"))

        year_graph = (yyyy | yyy | yy) + delete("年")
        year_only_graph = (yyyy | yyy | yy) + accep("年")
        self.year = year_graph
        self.year_only = year_only_graph
        self.month = mm
        self.day = dd

        year = self.tag_field("year", year_graph) + insert(" ")
        year_only = self.tag_field("year", year_only_graph) + insert(' preserve_order: "true"')
        month = self.tag_field("month", self.month)
        day = insert(" ") + self.tag_field("day", self.day)

        # yyyy/mm/dd | yyyy/mm | mm/dd | yyyy
        date = ((year + month + day) | (year + month) | (month + day)) | year_only
        self.tagger = self.add_tokens(date)

    def build_verbalizer(self):
        addsign = insert("/")
        year = self.verbalize_field("year", self.year) + delete(" ")
        year_only = self.verbalize_field("year", self.year_only) + delete(' preserve_order: "true"')
        month = self.verbalize_field("month", self.month)
        day = self.verbalize_field("day", self.day, leading_space=True)
        verbalizer = (year + addsign).ques + month + (addsign + day).ques
        verbalizer |= year_only
        self.verbalizer = self.delete_tokens(verbalizer)
