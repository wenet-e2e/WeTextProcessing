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

from tn.processor import Processor

from pynini import cross, accep, string_file
from pynini.lib.pynutil import delete, insert


class Cardinal(Processor):

    def __init__(self, enable_standalone_number=True):
        super().__init__('cardinal')
        self.number = None
        self.enable_standalone_number = enable_standalone_number
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        zero = string_file('itn/chinese/data/number/zero.tsv')    # 0
        digit = string_file('itn/chinese/data/number/digit.tsv')  # 1 ~ 9
        sign = string_file('itn/chinese/data/number/sign.tsv')    # + -
        dot = string_file('itn/chinese/data/number/dot.tsv')      # .

        addzero = insert('0')
        digits = zero | digit  # 0 ~ 9

        # 十一 => 11, 十二 => 12
        teen = cross('十', '1') + (digit | addzero)
        # 一十一 => 11, 二十一 => 21, 三十 => 30
        tens = digit + delete('十') + (digit | addzero)
        # 111, 101, 100
        hundred = (digit + delete('百') + (tens | (zero + digit) | addzero**2))
        # 1111, 1011, 1001, 1000
        thousand = (digit + delete('千') + (hundred
                                           | (zero + tens)
                                           | (addzero + zero + digit)
                                           | addzero**3))
        # 10001111, 1001111, 101111, 11111, 10111, 10011, 10001, 10000
        ten_thousand = ((thousand | hundred | teen | digit) + delete('万') +
                        (thousand
                         | (zero + hundred)
                         | (addzero + zero + tens)
                         | (addzero + addzero + zero + digit)
                         | addzero**4))

        # 1.11, 1.01
        number = digits | teen | tens | hundred | thousand | ten_thousand
        # 兆/亿
        number = ((number + accep('兆') + delete('零').ques).ques +
                  (number + accep('亿') + delete('零').ques).ques + number)
        number = sign.ques + number + (dot + digits.plus).ques
        self.number = number.optimize()
        self.digits = digits.optimize()

        # cardinal string like 127.0.0.1, used in ID, IP, etc.
        cardinal = digit.plus + (dot + digits).plus
        # float number like 1.11
        cardinal |= (number + dot + digits.plus)
        # cardinal string like 110 or 12306 or 13125617878, used in phone
        cardinal |= (digits**3 | digits**5 | digits**11)
        # cardinal string like 23
        if self.enable_standalone_number:
            cardinal |= number
        tagger = insert('value: "') + cardinal + insert('"')
        self.tagger = self.add_tokens(tagger)
