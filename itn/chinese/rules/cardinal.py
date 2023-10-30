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

    def __init__(self, enable_standalone_number=True, enable_0_to_9=True):
        super().__init__('cardinal')
        self.number = None
        self.number_exclude_0_to_9 = None
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
        teen = cross('十', '1') + (digit | add_weight(addzero, 0.1))
        # 一十一 => 11, 二十一 => 21, 三十 => 30
        tens = digit + delete('十') + (digit | add_weight(addzero, 0.1))
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
                    | add_weight(zero + (tens | teen), 0.1)
                    | add_weight(addzero + zero + digit, 0.5)
                    | add_weight(digit + addzero**2, 0.8)
                    | add_weight(addzero**3, 1.0)))
        # 10001111, 1001111, 101111, 11111, 10111, 10011, 10001, 10000
        ten_thousand = ((thousand | hundred | teen | tens | digits)
                        + delete('万')
                        + (thousand
                           | add_weight(zero + hundred, 0.1)
                           | add_weight(addzero + zero + (tens | teen), 0.5)
                           | add_weight(addzero + addzero + zero + digit, 0.5)
                           | add_weight(digit + addzero**3, 0.8)
                           | add_weight(addzero**4, 1.0)))
        # 个/十/百/千/万
        number = digits | teen | tens | hundred | thousand | ten_thousand
        # 兆/亿
        number = ((number + accep('兆') + delete('零').ques).ques +
                  (number + accep('亿') + delete('零').ques).ques + number)
        # 负的xxx 1.11, 1.01
        number = sign.ques + number + (dot + digits.plus).ques
        # 五六万，三五千，六七百，三四十
        special_2number = digit + insert("0~") + digit + cross("十", "0")
        special_2number |= digit + insert("00~") + digit + cross("百", "00")
        special_2number |= digit + insert("000~") + digit + cross("千", "000")
        special_2number |= digit + insert("0000~") + digit + cross("万", "0000")
        number |= special_2number
        # 十七八美元 => $17~18, 四十五六岁 => 45-6岁,
        # 三百七八公里 => 370-80km, 三百七八十千克 => 370-80kg
        special_3number = cross('十', '1') + digit + insert("~1") + digit
        special_3number |= digit + delete('十') + digit + insert("-") + digit
        special_3number |= digit + delete('百') + digit + insert("0-") + digit \
            + (insert("0") | add_weight(cross("十", "0"), -0.1))
        number |= add_weight(special_3number, -100.0)

        self.number = number.optimize()
        self.special_2number = special_2number.optimize()
        self.special_3number = special_3number.optimize()

        # 十/百/千/万
        number_exclude_0_to_9 = teen | tens | hundred | thousand | ten_thousand
        # 兆/亿
        number_exclude_0_to_9 = (
            (number_exclude_0_to_9 + accep('兆') + delete('零').ques).ques +
            (number_exclude_0_to_9 + accep('亿') + delete('零').ques).ques +
            number_exclude_0_to_9
        )
        # 负的xxx 1.11, 1.01
        number_exclude_0_to_9 |= (
            (number_exclude_0_to_9 | digits) +
            (dot + digits.plus).plus
        )
        # 五六万，三五千，六七百，三四十
        # 十七八美元 => $17~18, 四十五六岁 => 45-6岁,
        # 三百七八公里 => 370-80km, 三百七八十千克 => 370-80kg
        number_exclude_0_to_9 |= special_2number
        number_exclude_0_to_9 |= add_weight(special_3number, -100.0)

        self.number_exclude_0_to_9 = (sign.ques + number_exclude_0_to_9).optimize()  # noqa

        # cardinal string like 127.0.0.1, used in ID, IP, etc.
        cardinal = digit.plus + (dot + digits.plus).plus
        # float number like 1.11
        cardinal |= (number + dot + digits.plus)
        # cardinal string like 110 or 12306 or 13125617878, used in phone
        cardinal |= (digits**3 | digits**5 | digits**11)
        # cardinal string like 23
        if self.enable_standalone_number:
            if self.enable_0_to_9:
                cardinal |= number
            else:
                cardinal |= number_exclude_0_to_9
        tagger = insert('value: "') + cardinal + (insert(" ") + cardinal).star \
            + insert('"')
        self.tagger = self.add_tokens(tagger)
