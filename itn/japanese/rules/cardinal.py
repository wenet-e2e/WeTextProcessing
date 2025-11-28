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

from pynini import accep, cross, string_file
from pynini.lib.pynutil import delete, insert

from tn.processor import Processor
from tn.utils import get_abs_path


class Cardinal(Processor):

    def __init__(self, enable_standalone_number=True, enable_0_to_9=True, enable_million=False):
        super().__init__(name="cardinal")
        self.enable_standalone_number = enable_standalone_number
        self.enable_0_to_9 = enable_0_to_9
        self.enable_million = enable_million
        self.ten_thousand_minus = None  # used for year of date
        self.number = None
        self.number_exclude_0_to_9 = None
        self.decimal = None  # used for math
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        zero = string_file(get_abs_path("../itn/japanese/data/number/zero.tsv"))
        digit = string_file(get_abs_path("../itn/japanese/data/number/digit.tsv"))
        hundred_digit = string_file(get_abs_path("../itn/japanese/data/number/hundred_digit.tsv"))
        sign = string_file(get_abs_path("../itn/japanese/data/number/sign.tsv"))
        dot = string_file(get_abs_path("../itn/japanese/data/number/dot.tsv"))
        ties = string_file(get_abs_path("../itn/japanese/data/number/ties.tsv"))
        graph_teen = string_file(get_abs_path("../itn/japanese/data/number/teen.tsv"))

        addzero = insert("0")
        # 〇 三 九
        digits = zero | digit  # 0 ~ 9
        # 十 十一 十九
        teen = graph_teen
        teen |= cross("十", "1") + (digit | addzero)
        # 三十 三十一 九十 九十一
        tens = ties + addzero | (ties + (digit | addzero))

        # 三百二 三百
        hundred = digit + delete("百") + (tens | teen | addzero**1 + digits | addzero**2)
        # 百 百十 百二十三
        hundred |= cross("百", "1") + (tens | teen | addzero + digits | addzero**2)
        # 百一 百二
        hundred |= hundred_digit

        # 二千百 二千三百 二千百一 九千百二十三 九千二十三 九千二十 九千二
        thousand = (
            (hundred | teen | tens | digits)
            + delete("千")
            + (hundred | addzero + tens | addzero + teen | addzero**2 + digits | addzero**3)
        )
        #  千百 千三百 千百一 千百二十三 千二十三 千二十 千二
        thousand |= cross("千", "1") + (hundred | addzero + tens | addzero + teen | addzero**2 + digits | addzero**3)

        # 一万 二万二 二万二千百 二万二千三百 一万二千百一 一万九千百二十三 一万九千二十三 九万九千二十 四万九千二
        if self.enable_million:
            ten_thousand = (
                (thousand | hundred | teen | tens | digits)
                + delete("万")
                + (
                    thousand
                    | addzero + hundred
                    | addzero**2 + tens
                    | addzero**2 + teen
                    | addzero**3 + digits
                    | addzero**4
                )
            )
        else:
            # 二万 十万 三十四万
            ten_thousand = (
                (teen | tens | digits)
                + delete("万")
                + (
                    thousand
                    | addzero + hundred
                    | addzero**2 + tens
                    | addzero**2 + teen
                    | addzero**3 + digits
                    | addzero**4
                )
            )
            # 三百四十万 三千四百万
            ten_thousand |= (thousand | hundred) + accep("万") + (thousand | hundred | tens | teen | digits).ques

        # 0~9999
        ten_thousand_minus = digits | teen | tens | hundred | thousand
        self.ten_thousand_minus = ten_thousand_minus

        # 0~99999999
        positive_integer = ten_thousand_minus | ten_thousand

        # ±0~99999999
        number = (sign.ques + positive_integer).optimize()
        self.number = number
        self.number_exclude_0_to_9 = teen | tens | hundred | thousand | ten_thousand

        # ±0.0~99999999.99...
        decimal = sign.ques + positive_integer + dot + digits.plus
        self.decimal = decimal
        # % like -27.00%
        percent = (number | decimal) + cross("パーセント", "%")

        # ±100,000,000~∞  e.g. 一兆三百二十万五千 => 1兆320万5000
        big_integer = sign.ques + (
            (
                (ten_thousand_minus + accep("兆")).ques
                + ten_thousand_minus
                + accep("億")
                + ten_thousand_minus
                + accep("万").ques
                + ten_thousand_minus.ques
            )
            | (
                ten_thousand_minus
                + accep("兆")
                + (ten_thousand_minus + accep("億")).ques
                + ten_thousand_minus
                + accep("万").ques
                + ten_thousand_minus.ques
            )
        )
        self.big_integer = number | big_integer

        # cardinal string like 127.0.0.1, used in ID, IP, etc.
        cardinal = digit.plus + (dot + digits.plus).plus
        # float number like 1.11
        cardinal |= decimal
        # cardinal string like 110 or 12306 or 13125617878, used in phone
        cardinal |= digits**2 + digits.plus
        # % like -27.00%
        cardinal |= percent

        # allow convert standalone number
        if self.enable_standalone_number:
            if self.enable_0_to_9:
                # 一 => 1    四 => 4    一秒 => 1秒   一万二 => 12000 二十三 => 23
                cardinal |= number | big_integer
            else:
                # 一 => 一   四 => 四   一秒 => 1秒   一万二 => 一万二 二三 => 23
                number_two_plus = sign.ques + (
                    (digits + digits.plus) | teen | tens | hundred | thousand | ten_thousand | big_integer
                )

                cardinal |= number_two_plus

        self.tagger = self.add_tokens(insert('value: "') + cardinal + insert('"')).optimize()
