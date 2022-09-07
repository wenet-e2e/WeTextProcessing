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

from tn.chinese.rules.cardinal import Cardinal
from tn.processor import Processor

from pynini import cross, string_file
from pynini.lib.pynutil import delete, insert


class Measure(Processor):

    def __init__(self):
        super().__init__(name='measure')
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        units_en = string_file('tn/chinese/data/measure/units_en.tsv')
        units_zh = string_file('tn/chinese/data/measure/units_zh.tsv')
        units = units_en | units_zh
        rmspace = delete(' ').ques

        number = Cardinal().number
        percent = insert('百分之') + number + delete('%')

        number @= self.build_rule(cross('二', '两'), '[BOS]', '[EOS]')
        measure = number + rmspace + units
        measure @= self.build_rule(cross('两两', '二两'), '[BOS]', '')
        tagger = insert('value: "') + (measure | percent) + insert('"')

        # 10km/h
        rmsign = rmspace + delete('/') + rmspace
        tagger |= (
            insert('numerator: "') + measure + rmsign +
            insert('" denominator: "') + units + insert('"'))
        self.tagger = self.add_tokens(tagger)

    def build_verbalizer(self):
        super().build_verbalizer()
        denominator = delete('denominator: "') + self.SIGMA + delete('" ')
        numerator = delete('numerator: "') + self.SIGMA + delete('"')
        verbalizer = insert('每') + denominator + numerator
        self.verbalizer |= self.delete_tokens(verbalizer)
