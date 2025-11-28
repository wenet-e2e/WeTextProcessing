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

    def __init__(self):
        super().__init__(name='date')
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        digit = string_file(get_abs_path('chinese/data/number/digit.tsv'))
        zero = string_file(get_abs_path('chinese/data/number/zero.tsv'))

        yyyy = digit + (digit | zero)**3
        m = string_file(get_abs_path('chinese/data/date/m.tsv'))
        mm = string_file(get_abs_path('chinese/data/date/mm.tsv'))
        d = string_file(get_abs_path('chinese/data/date/d.tsv'))
        dd = string_file(get_abs_path('chinese/data/date/dd.tsv'))
        rmsign = (delete('/') | delete('-') | delete('.')) + insert(' ')

        year = insert('year: "') + yyyy + insert('年"')
        month = insert('month: "') + (m | mm) + insert('"')
        day = insert('day: "') + (d | dd) + insert('"')

        # yyyy/m/d | yyyy/mm/dd | dd/mm/yyyy
        # yyyy/0m | 0m/yyyy | 0m/dd
        mm = insert('month: "') + mm + insert('"')
        date = ((year + rmsign + month + rmsign + day)
                | (day + rmsign + month + rmsign + year)
                | (year + rmsign + mm)
                | (mm + rmsign + year)
                | (mm + rmsign + day))
        tagger = self.add_tokens(date)

        to = (delete('-') | delete('~')) + insert(' char { value: "到" } ')
        self.tagger = tagger + (to + tagger).ques

    def build_verbalizer(self):
        year = delete('year: "') + self.SIGMA + delete('" ')
        month = delete('month: "') + self.SIGMA + delete('"')
        day = delete(' day: "') + self.SIGMA + delete('"')
        verbalizer = year.ques + month + day.ques
        self.verbalizer = self.delete_tokens(verbalizer)
