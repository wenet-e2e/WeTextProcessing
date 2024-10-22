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

from tn.processor import Processor
from tn.utils import get_abs_path

from pynini import accep, cross, string_file
from pynini.lib.pynutil import delete, insert


class Cardinal(Processor):

    def __init__(self,
                 enable_standalone_number=True,
                 enable_0_to_9=True,
                 enable_million=False):
        super().__init__(name='cardinal')
        self.enable_standalone_number = enable_standalone_number
        self.enable_0_to_9 = enable_0_to_9
        self.enable_million = enable_million
        self.ten_thousand_minus = None  # used for year of date
        self.positive_integer = None  # used for ordinal、measure
        self.big_integer = None  # for math
        self.number = None
        self.decimal = None  # used for math
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        zero = string_file(
            get_abs_path("../itn/japanese/data/number/zero.tsv"))
        digit = string_file(
            get_abs_path("../itn/japanese/data/number/digit.tsv"))
        hundred_digit = string_file(
            get_abs_path("../itn/japanese/data/number/hundred_digit.tsv"))
        sign = string_file(
            get_abs_path("../itn/japanese/data/number/sign.tsv"))
        dot = string_file(get_abs_path("../itn/japanese/data/number/dot.tsv"))
        ties = string_file(
            get_abs_path("../itn/japanese/data/number/ties.tsv"))
        graph_teen = string_file(
            get_abs_path("../itn/japanese/data/number/teen.tsv"))

        addzero = insert("0")
        # 〇 三 九
        digits = zero | digit  # 0 ~ 9
        # 十 十一 十九
        teen = graph_teen
        teen |= cross("十", "1") + (digit | addzero)
        # 三十 三十一 九十 九十一
        tens = ties + addzero | (ties + (digit | addzero))

        # 三百二 三百
        hundred = (digit + delete("百") +
                   (tens | teen | addzero**1 + digits | addzero**2))
        # 百 百十 百二十三
        hundred |= cross("百",
                         "1") + (tens | teen | addzero + digits | addzero**2)
        # 百一 百二
        hundred |= hundred_digit

        # 二千百 二千三百 二千百一 九千百二十三 九千二十三 九千二十 九千二
        thousand = ((hundred | teen | tens | digits) + delete("千") +
                    (hundred | addzero + tens | addzero + teen
                     | addzero**2 + digits | addzero**3))
        #  千百 千三百 千百一 千百二十三 千二十三 千二十 千二
        thousand |= cross('千', '1') + (hundred | addzero + tens | addzero +
                                       teen | addzero**2 + digits | addzero**3)

        # 一万 二万二 二万二千百 二万二千三百 一万二千百一 一万九千百二十三 一万九千二十三 九万九千二十 四万九千二
        ten_thousand = ((thousand | hundred | teen | tens | digits) +
                        delete("万") +
                        (thousand | addzero + hundred | addzero**2 + tens |
                         addzero**2 + teen | addzero**3 + digits | addzero**4))

        hundred_thousand = (digits + delete("十万") + (ten_thousand
                                                     | addzero + thousand
                                                     | addzero**2 + hundred
                                                     | addzero**3 + tens
                                                     | addzero**3 + teen
                                                     | addzero**4 + digits
                                                     | addzero**5))

        million = (digits + delete("百万") + (hundred_thousand
                                            | addzero + ten_thousand
                                            | addzero**2 + thousand
                                            | addzero**3 + hundred
                                            | addzero**4 + tens
                                            | addzero**4 + teen
                                            | addzero**5 + digits
                                            | addzero**6))

        ten_million = (digits + delete("千万") + (million
                                                | addzero + hundred_thousand
                                                | addzero**2 + ten_thousand
                                                | addzero**3 + thousand
                                                | addzero**4 + hundred
                                                | addzero**5 + tens
                                                | addzero**5 + teen
                                                | addzero**6 + digits
                                                | addzero**7))

        # 0~9999
        ten_thousand_minus = digits | teen | tens | hundred | thousand
        self.ten_thousand_minus = ten_thousand_minus
        # 0~99999999
        positive_integer = (digits | teen | tens | hundred | thousand
                            | ten_thousand | hundred_thousand | million
                            | ten_million)
        self.positive_integer = positive_integer

        # ±0~9999
        number = (sign.ques + ten_thousand_minus).optimize()
        self.number = number

        # ±0.0~99999999.99...
        decimal = sign.ques + positive_integer + dot + digits.plus
        self.decimal = decimal

        if self.enable_million:
            # 三百二十万五千 => 3255000
            number = (sign.ques + positive_integer).optimize()

        # ±10000~∞  e.g. 一兆三百二十万五千 => 1兆320万5000
        big_integer = (sign.ques + (
            (ten_thousand_minus + accep("兆")).ques +
            (ten_thousand_minus + accep("億")).ques +
            (ten_thousand_minus + accep("万")).ques + ten_thousand_minus
            | (ten_thousand_minus + accep("兆")).ques +
            (ten_thousand_minus + accep("億")).ques +
            (ten_thousand_minus + accep("万"))
            | (ten_thousand_minus + accep("兆")).ques +
            (ten_thousand_minus + accep("億"))
            | ten_thousand_minus + accep("兆")
            | ten_thousand_minus))
        self.big_integer = big_integer
        number |= big_integer

        # cardinal string like 127.0.0.1, used in ID, IP, etc.
        cardinal = digit.plus + (dot + digits.plus).plus
        # float number like 1.11
        cardinal |= decimal
        # cardinal string like 110 or 12306 or 13125617878, used in phone
        cardinal |= digits**3 | digits**5 | digits**10 | digits**11 | digits**12

        # allow convert standalone number
        if self.enable_standalone_number:
            if self.enable_0_to_9:
                # 一 => 1    四 => 4    一秒 => 1秒   一万二 => 1万2 二三 => 23
                cardinal |= number
            else:
                # 一 => 一   四 => 四   一秒 => 1秒   一万二 => 一万二 二三 => 23
                number_two_plus = ((digits + digits.plus)
                                   | teen
                                   | tens
                                   | hundred
                                   | thousand)
                cardinal |= number_two_plus

        self.tagger = self.add_tokens(
            insert('value: "') + cardinal + insert('"')).optimize()
