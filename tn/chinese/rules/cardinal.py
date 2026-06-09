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

from pynini import accep, cross, string_file
from pynini.lib.pynutil import add_weight, delete, insert

from tn.processor import Processor
from tn.utils import get_abs_path


class Cardinal(Processor):

    def __init__(self):
        super().__init__("cardinal")
        self.number = None
        self.digits = None
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        zero = string_file(get_abs_path("chinese/data/number/zero.tsv"))
        digit = string_file(get_abs_path("chinese/data/number/digit.tsv"))
        teen = string_file(get_abs_path("chinese/data/number/teen.tsv"))
        sign = string_file(get_abs_path("chinese/data/number/sign.tsv"))
        dot = string_file(get_abs_path("chinese/data/number/dot.tsv"))

        rmzero = delete("0") | delete("０")
        digits = zero | digit
        self.digits = digits

        # 11 => 十一, 10 => 十
        ten = teen + insert("十") + (digit | rmzero)
        # 21 => 二十一, 30 => 三十
        tens = digit + insert("十") + (digit | rmzero)
        # 111 => 一百一十一, 101 => 一百零一, 100 => 一百
        hundred = digit + insert("百") + (tens | (zero + digit) | rmzero**2)
        # 1111 => 一千一百一十一, 1001 => 一千零一, 1000 => 一千
        thousand = digit + insert("千") + (hundred | (zero + tens) | (rmzero + zero + digit) | rmzero**3)

        # exactly 4 input digits, not all zero
        four_nonzero = thousand | (zero + hundred) | (rmzero + zero + tens) | (rmzero**2 + zero + digit)
        # exactly 4 input digits, may be all zero
        four_any = four_nonzero | rmzero**4

        # 万级: 1,0000 ~ 9999,9999 (5~8位)
        ten_thousand = (
            (thousand | hundred | ten | digit)
            + insert("万")
            + four_any
        )

        # 亿级: 1,0000,0000 ~ 9999,9999,9999 (9~13位)
        hundred_million = (
            (thousand | hundred | ten | digit)
            + insert("亿")
            + (four_nonzero + insert("万") + four_any | rmzero**4 + four_nonzero | rmzero**8)
        )

        number = digits | ten | hundred | thousand | ten_thousand | hundred_million
        number = sign.ques + number + (dot + digits.plus).ques
        number @= self.build_rule(
            cross("二百", "两百") | cross("二千", "两千") | cross("二万", "两万") | cross("二亿", "两亿"), "[BOS]"
        ).optimize()
        percent = insert("百分之") + number + delete("%")
        self.number = accep("约").ques + accep("人均").ques + (number | percent)

        # cardinal string like 127.0.0.1, used in ID, IP, etc.
        cardinal = digits.plus + (dot + digits.plus) ** 3
        cardinal |= percent
        # xxxx-xxx-xxx
        cardinal |= digits.plus + (delete("-") + digits.plus) ** 2
        # xxx-xxxxxxxx
        cardinal |= digits**3 + delete("-") + digits**8
        # three or five or eleven phone numbers
        phone_digits = digits @ self.build_rule(cross("一", "幺"))
        phone = phone_digits**3 | phone_digits**5 | phone_digits**11
        phone |= accep("尾号") + (accep("是") | accep("为")).ques + phone_digits**4
        cardinal |= add_weight(phone, -1.0)

        tagger = insert('value: "') + cardinal + insert('"')
        self.tagger = self.add_tokens(tagger)
