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

from pynini import string_file
from pynini.lib.pynutil import delete, insert


class Measure(Processor):

    def __init__(self):
        super().__init__(name='measure')
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        units_en = string_file('itn/chinese/data/measure/units_en.tsv')
        units_zh = string_file('itn/chinese/data/measure/units_zh.tsv')
        sign = string_file('itn/chinese/data/number/sign.tsv')    # + -
        units = units_en | units_zh

        number = Cardinal().number
        percent = ((sign + delete('的').ques).ques + delete('百分之') +
                   number + insert('%'))

        # 十千米每小时 => 10km/h
        measure = number + units
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
