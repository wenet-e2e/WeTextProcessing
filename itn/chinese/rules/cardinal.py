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

from pynini import accep, cross, string_file
from pynini.lib.pynutil import add_weight, delete, insert

from tn.processor import Processor
from tn.utils import get_abs_path


class Cardinal(Processor):

    def __init__(self, enable_standalone_number=True, enable_0_to_9=True, enable_million=False):
        super().__init__("cardinal")
        self.number = None
        self.number_exclude_0_to_9 = None
        self.enable_standalone_number = enable_standalone_number
        self.enable_0_to_9 = enable_0_to_9
        self.enable_million = enable_million
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        zero = string_file(get_abs_path("../itn/chinese/data/number/zero.tsv"))  # 0
        digit = string_file(get_abs_path("../itn/chinese/data/number/digit.tsv"))  # 1 ~ 9
        special_tilde = string_file(get_abs_path("../itn/chinese/data/number/special_tilde.tsv"))  # 七八十->70~80
        special_tilde = special_tilde + add_weight((accep("万") | accep("亿")), -0.1).ques
        special_dash = string_file(get_abs_path("../itn/chinese/data/number/special_dash.tsv"))  # 七八十->70-80
        special_dash = special_dash + add_weight((accep("万") | accep("亿")), -0.1).ques
        sign = string_file(get_abs_path("../itn/chinese/data/number/sign.tsv"))  # + -
        dot = string_file(get_abs_path("../itn/chinese/data/number/dot.tsv"))  # .

        # 0. 基础数字
        addzero = insert("0")
        digits = zero | digit  # 0 ~ 9
        # 十一 => 11, 十二 => 12
        teen = cross("十", "1") + (digit | add_weight(addzero, 0.1))
        # 一十一 => 11, 二十一 => 21, 三十 => 30
        tens = digit + delete("十") + (digit | add_weight(addzero, 0.1))
        # 一百一十 => 110, 一百零一 => 101, 一百一 => 110, 一百 => 100
        hundred = (
            digit
            + delete("百")
            + (
                tens
                | teen
                | add_weight(zero + digit, 0.1)
                | add_weight(digit + addzero, 0.5)
                | add_weight(addzero**2, 1.0)
            )
        )
        # 一千一百一十一 => 1111, 一千零一十一 => 1011, 一千零一 => 1001
        # 一千一 => 1100, 一千 => 1000
        thousand = (
            digit
            + delete("千")
            + (
                hundred
                | add_weight(zero + (tens | teen), 0.1)
                | add_weight(addzero + zero + digit, 0.5)
                | add_weight(digit + addzero**2, 0.8)
                | add_weight(addzero**3, 1.0)
            )
        )
        # 10001111, 1001111, 101111, 11111, 10111, 10011, 10001, 10000
        if self.enable_million:
            ten_thousand = (
                (thousand | hundred | teen | tens | digit)
                + delete("万")
                + (
                    thousand
                    | add_weight(zero + hundred, 0.1)
                    | add_weight(addzero + zero + (tens | teen), 0.5)
                    | add_weight(addzero + addzero + zero + digit, 0.5)
                    | add_weight(digit + addzero**3, 0.8)
                    | add_weight(addzero**4, 1.0)
                )
            )
        else:
            ten_thousand = (
                (teen | tens | digit)
                + delete("万")
                + (
                    thousand
                    | add_weight(zero + hundred, 0.1)
                    | add_weight(addzero + zero + (tens | teen), 0.5)
                    | add_weight(addzero + addzero + zero + digit, 0.5)
                    | add_weight(digit + addzero**3, 0.8)
                    | add_weight(addzero**4, 1.0)
                )
            )
            ten_thousand |= (
                (thousand | hundred)
                + accep("万")
                + delete("零").ques
                + (thousand | hundred | tens | teen | digits).ques
            )

        # 1. 利用基础数字所构建的包含0~9的标准数字
        # 个/十/百/千/万
        number = digits | teen | tens | hundred | thousand | ten_thousand
        # 兆/亿
        number = (
            (number + accep("兆") + delete("零").ques).ques + (number + accep("亿") + delete("零").ques).ques + number
        )
        # 负的xxx 1.11, 1.01
        number = sign.ques + number + (dot + digits.plus).ques
        # 五六万 => 5~6万，三五千 => 3000~5000，六七百 => 600~700，三四十 => 30~40, 三四十亿 => 30~40亿
        number |= special_tilde
        # 十七八 => 17-8, 四十五六 => 45-6, 三百七八十 => 370-80, 四十五六万 => 45-6万, 一万六七 => 16000-7000
        _special_dash = cross("十", "1") + special_dash
        _special_dash |= digit + delete("十") + special_dash
        _special_dash |= digit + delete("百") + special_dash
        _special_dash |= digit + delete("万") + digit + insert("000-") + digit + insert("000")
        number |= _special_dash

        self.number = number.optimize()
        self.special_tilde = special_tilde.optimize()
        self.special_dash = _special_dash.optimize()

        # 2. 利用基础数字所构建的不包含0~9的标准数字
        # 十/百/千/万
        number_exclude_0_to_9 = teen | tens | hundred | thousand | ten_thousand
        # 兆/亿
        number_exclude_0_to_9 = (
            ((number_exclude_0_to_9 | digits) + accep("兆") + delete("零").ques).ques
            + ((number_exclude_0_to_9 | digits) + accep("亿") + delete("零").ques).ques
            + number_exclude_0_to_9
        )
        # 负的xxx 1.11, 1.01
        number_exclude_0_to_9 |= (number_exclude_0_to_9 | digits) + (dot + digits.plus).plus
        # 五六万，三五千，六七百，三四十
        # 十七八美元 => $17~18, 四十五六岁 => 45-6岁,
        # 三百七八公里 => 370-80km, 三百七八十千克 => 370-80kg
        number_exclude_0_to_9 |= special_tilde
        number_exclude_0_to_9 |= add_weight(_special_dash, -0.1)

        self.number_exclude_0_to_9 = (sign.ques + number_exclude_0_to_9).optimize()

        # 3. 特殊格式的数字
        # cardinal string like 127.0.0.1, used in ID, IP, etc.
        cardinal = digits.plus + (dot + digits.plus).plus
        # float number like 1.11
        cardinal |= number + dot + digits.plus
        # cardinal string like 110 or 12306 or 13125617878, used in phone,
        #   340621199806051223, used in ID card
        idcard_last_char = digits | "X" | "x"
        cardinal |= digits**3 | digits**4 | digits**5 | digits**11 | (digits**17 + idcard_last_char) | digits**18

        # 4. 特殊格式的数字 + 标准数字
        # cardinal string like 23
        if self.enable_standalone_number:
            if self.enable_0_to_9:
                # 特殊格式数字为第一优先级, 标准数字为第二优先级, 如 "一二三四"
                # 优先转译为 "1234" 而非 "1~2 3~4"
                cardinal |= add_weight(number, 0.1)
            else:
                cardinal |= add_weight(number_exclude_0_to_9, 0.1)

        
        # 5. 添加"中文数字+英文字母"的规则，如"四a" -> "4a"
        # 匹配一个或多个英文字母（大小写）
        from pynini import union
        english_letters = union(*[accep(c) for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"])
        # 数字+字母的组合，如"四a" -> "4a"
        number_with_letter = number + english_letters.plus
        cardinal |= add_weight(number_with_letter, 0.05)  # 使用较高优先级

        # 6. 添加两个连续完整数字的范围规则（如"二十一二十二" -> "21-22"）
        # 定义完整数字（不包括单个数字0-9，避免误匹配）
        complete_number = teen | tens | hundred | thousand | ten_thousand
        complete_number = (
            (complete_number + accep("兆") + delete("零").ques).ques 
            + (complete_number + accep("亿") + delete("零").ques).ques 
            + complete_number
        )
        complete_number = sign.ques + complete_number + (dot + digits.plus).ques
        
        # 两个连续完整数字的范围模式（优先级高于单独的数字）
        # 如：二十一二十二 -> 21-22, 三十一三十二 -> 31-32
        number_range = complete_number + insert("~") + complete_number
        # 将这个规则添加到 cardinal，使用较高优先级（负权重）
        cardinal |= add_weight(number_range, -0.05)
        
        tagger = insert('value: "') + cardinal + (insert(" ") + cardinal).star + insert('"')
        self.tagger = self.add_tokens(tagger)
