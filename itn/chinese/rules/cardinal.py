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
from pynini.lib.pynutil import delete, insert, add_weight


class Cardinal(Processor):

    def __init__(self,
        enable_standalone_number=True,
        enable_0_to_9=False):
        super().__init__('cardinal')
        self.number = None
        self.enable_standalone_number = enable_standalone_number
        self.enable_0_to_9 = enable_0_to_9
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
        # 一百一十 => 110, 一百零一 => 101, 一百一 => 110, 一百 => 100
        hundred = (digit + delete('百') + (tens
                                          | teen
                                          | add_weight(zero + digit, 0.1)
                                          | add_weight(digit + addzero, 0.5)
                                          | add_weight(addzero**2, 1.0)))
        # 一千一百一十一 => 1111, 一千零一十一 => 1011, 一千零一 => 1001
        # 一千一 => 1100, 一千 => 1000
        thousand = ((hundred | teen | tens | digits) + delete('千') + (
                    hundred
                    | add_weight(zero + tens, 0.1)
                    | add_weight(addzero + zero + digit, 0.5)
                    | add_weight(digit + addzero**2, 0.8)
                    | add_weight(addzero**3, 1.0)))
        # 10001111, 1001111, 101111, 11111, 10111, 10011, 10001, 10000
        ten_thousand = ((thousand | hundred | teen | tens | digits)
                        + delete('万')
                        + (thousand
                           | add_weight(zero + hundred, 0.1)
                           | add_weight(addzero + zero + tens, 0.5)
                           | add_weight(addzero + addzero + zero + digit, 0.5)
                           | add_weight(digit + addzero**3, 0.8)
                           | add_weight(addzero**4, 1.0)))

        # 1.11, 1.01
        number = digits | teen | tens | hundred | thousand | ten_thousand
        # 兆/亿
        number = ((number + accep('兆') + delete('零').ques).ques +
                  (number + accep('亿') + delete('零').ques).ques + number)
        number = sign.ques + number + (dot + digits.plus).ques
        self.number = number.optimize()
        self.digits = digits.optimize()

        # cardinal string like 127.0.0.1, used in ID, IP, etc.
        cardinal = digit.plus + (dot + digits.plus).plus
        # float number like 1.11
        cardinal |= (number + dot + digits.plus)
        # cardinal string like 110 or 12306 or 13125617878, used in phone
        cardinal |= (digits**3 | digits**5 | digits**11)
        # cardinal string like 23
        if self.enable_standalone_number:
            #  number_two_plus = number + number.plus
            if self.enable_0_to_9:
                cardinal |= number
            else:
                number_two_plus = (digits + digits.plus) | teen | tens | hundred | thousand | ten_thousand
                cardinal |= number_two_plus
        tagger = insert('value: "') + cardinal + insert('"')
        self.tagger = self.add_tokens(tagger)
