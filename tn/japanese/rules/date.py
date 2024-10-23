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

from tn.japanese.rules.cardinal import Cardinal
from tn.processor import Processor
from tn.utils import get_abs_path

from pynini import string_file
from pynini.lib.pynutil import delete, insert


class Date(Processor):
    """
    1日 -> 一日
    5~9日 -> 五から九日
    1月 -> 一月
    3~4月 -> 三から四月
    1月1日 -> 一月一日
    70年代 -> 七十年代
    70~80年代 -> 七十から八十年代
    21世紀 -> 二十一世紀
    2009年 -> 二千九年
    23年2月25日(土) -> 二十三年二月二十五日土曜日
    7月5〜9日(月〜金) -> 七月五から九日月曜日から金曜日
    今年は令和6 -> 今年はR六
    2019年3月2日 -> 二千十九年三月二日
    1986年8月18日 -> 千九百八十六年八月十八日
    ---以上都可以把日期做为普通数字进行转换，以下要添加对应规则---
    2008/08/08 -> 二千八年八月八日
    1008/08/10 -> 千八年八月十日
    1108/08/10 -> 千百八年八月十日
    1000.08.10 -> 千年八月十日
    """

    def __init__(self):
        super().__init__(name='date')
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        yyyy = Cardinal().thousand
        m = string_file(get_abs_path('japanese/data/date/m.tsv'))
        mm = string_file(get_abs_path('japanese/data/date/mm.tsv'))
        d = string_file(get_abs_path('japanese/data/date/d.tsv'))
        dd = string_file(get_abs_path('japanese/data/date/dd.tsv'))
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
        # yyyy/0m | 0m/yyyy | 0m/dd
        simple_date = ((year + rmsign + month)
                       | (month + rmsign + year)
                       | (month + rmsign + day))

        tagger = self.add_tokens(date)
        simple_tagger = self.add_tokens(simple_date)

        to = ((delete('-') | delete('~') | delete('から')) +
              insert(' char { value: "から" } '))
        self.tagger = (tagger + (to + tagger).ques
                       | simple_tagger + to + simple_tagger)

    def build_verbalizer(self):
        year = delete('year: "') + self.SIGMA + delete('" ')
        month = delete('month: "') + self.SIGMA + delete('"')
        day = delete(' day: "') + self.SIGMA + delete('"')
        verbalizer = year.ques + month + day.ques
        self.verbalizer = self.delete_tokens(verbalizer)
