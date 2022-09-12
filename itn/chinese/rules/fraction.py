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

from pynini.lib.pynutil import delete, insert


class Fraction(Processor):

    def __init__(self):
        super().__init__(name='fraction')
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        number = Cardinal().number

        tagger = (insert('denominator: "') + number +
                  delete('分之') + insert('" numerator: "') +
                  number + insert('"')).optimize()
        self.tagger = self.add_tokens(tagger)

    def build_verbalizer(self):
        numerator = delete('numerator: "') + self.SIGMA + delete('"')
        denominator = delete(' denominator: "') + self.SIGMA + delete('"')
        verbalizer = numerator + insert('/') + denominator
        self.verbalizer = self.delete_tokens(verbalizer)
