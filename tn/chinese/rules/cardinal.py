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

from tn.processor import Processor

from pynini import accep, cross, string_file
from pynini.lib.pynutil import add_weight, delete, insert


class Cardinal(Processor):

    def __init__(self):
        super().__init__('cardinal')
        self.number = None
        self.digits = None
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        zero = string_file('tn/chinese/data/number/zero.tsv')
        digit = string_file('tn/chinese/data/number/digit.tsv')
        teen = string_file('tn/chinese/data/number/teen.tsv')
        sign = string_file('tn/chinese/data/number/sign.tsv')
        dot = string_file('tn/chinese/data/number/dot.tsv')

        rmzero = delete('0') | delete('０')
        rmpunct = delete(',').ques
        digits = zero | digit
        self.digits = digits

        # 11 => 十一
        ten = teen + insert('十') + (digit | rmzero)
        # 11 => 一十一
        tens = digit + insert('十') + (digit | rmzero)
        # 111, 101, 100
        hundred = (digit + insert('百') + (tens | (zero + digit) | rmzero**2))
        # 1111, 1011, 1001, 1000
        thousand = (digit + insert('千') + rmpunct + (hundred
                                                     | (zero + tens)
                                                     | (rmzero + zero + digit)
                                                     | rmzero**3))
        # 10001111, 1001111, 101111, 11111, 10111, 10011, 10001, 10000
        ten_thousand = ((thousand | hundred | ten | digit) + insert('万') +
                        (thousand
                         | (zero + rmpunct + hundred)
                         | (rmzero + rmpunct + zero + tens)
                         | (rmzero + rmpunct + rmzero + zero + digit)
                         | rmzero**4))

        # 1.11, 1.01
        number = digits | ten | hundred | thousand | ten_thousand
        number = sign.ques + number + (dot + digits.plus).ques
        number @= self.build_rule(
            cross('二百', '两百')
            | cross('二千', '两千')
            | cross('二万', '两万'))
        self.number = accep('约').ques + accep('人均').ques + number.optimize()

        # cardinal string like 127.0.0.1, used in ID, IP, etc.
        cardinal = digits.plus + (dot + digits.plus)**3
        # xxxx-xxx-xxx
        cardinal |= digits.plus + (delete('-') + digits.plus)**2
        # xxx-xxxxxxxx
        cardinal |= digits**3 + delete('-') + digits**8
        # three or five or eleven phone numbers
        phone_digits = digits @ self.build_rule(cross('一', '幺'))
        phone = phone_digits**3 | phone_digits**5 | phone_digits**11
        cardinal |= add_weight(phone, -1.0)

        tagger = insert('value: "') + cardinal + insert('"')
        self.tagger = self.add_tokens(tagger)
