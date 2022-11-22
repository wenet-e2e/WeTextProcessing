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

from itn.chinese.rules.cardinal import Cardinal
from tn.processor import Processor

from pynini import string_file, accep, cross
from pynini.lib.pynutil import delete, insert, add_weight


class Measure(Processor):

    def __init__(self, exclude_one=True):
        super().__init__(name='measure')
        self.exclude_one = exclude_one
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        units_en = string_file('itn/chinese/data/measure/units_en.tsv')
        units_zh = string_file('itn/chinese/data/measure/units_zh.tsv')
        sign = string_file('itn/chinese/data/number/sign.tsv')    # + -
        units = units_en | ((accep('亿') | accep('兆') | accep('万')).ques
                            + units_zh)

        number = Cardinal().number
        # 百分之三十, 百分三十, 百分之百
        percent = ((sign + delete('的').ques).ques + delete('百分') +
                   delete('之').ques + (number | cross('百', '100'))
                   + insert('%'))

        # 十千米每小时 => 10km/h
        measure = number + units
        if self.exclude_one:
            measure |= number + number.plus + units
            measure |= (add_weight(accep('一'), -1.0) + units_zh)
        tagger = insert('value: "') + (measure | percent) + insert('"')

        # 每小时十千米 => 10km/h
        tagger |= (
            insert('denominator: "') + delete('每') + units +
            insert('" numerator: "') + measure + insert('"'))
        self.tagger = self.add_tokens(tagger)

    def build_verbalizer(self):
        super().build_verbalizer()
        numerator = delete('numerator: "') + self.SIGMA + delete('"')
        denominator = delete(' denominator: "') + self.SIGMA + delete('"')
        verbalizer = numerator + insert('/') + denominator
        self.verbalizer |= self.delete_tokens(verbalizer)
