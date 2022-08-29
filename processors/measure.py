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

from processors.cardinal import Cardinal
from processors.processor import Processor

from pynini import cross, string_file
from pynini.lib.pynutil import delete, insert


class Measure(Processor):

    def __init__(self):
        super().__init__(name='measure')
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        units_en = string_file('data/measure/units_en.tsv')
        units_zh = string_file('data/measure/units_zh.tsv')

        number = Cardinal().number
        percent = insert('百分之') + number + delete('%')

        number @= self.build_rule(cross('二', '两'), '[BOS]', '[EOS]')
        measure = number + delete(' ').ques + (units_en | units_zh)
        measure @= self.build_rule(cross('两两', '二两'), '[BOS]', '[EOS]')

        tagger = percent | measure
        tagger = insert('value: "') + tagger + insert('"')
        self.tagger = self.add_tokens(tagger)
