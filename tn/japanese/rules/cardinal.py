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

from pynini import string_file, cross
from pynini.lib.pynutil import delete, insert


class Cardinal(Processor):

    def __init__(self):
        super().__init__(name='cardinal')
        self.thousand = None
        self.thousands = None
        self.number = None
        self.digits = None
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        zero = string_file(get_abs_path('japanese/data/number/zero.tsv'))
        # 1-9
        digit = string_file(get_abs_path('japanese/data/number/digit.tsv'))
        en_digit = string_file(
            get_abs_path('japanese/data/number/en_digit.tsv'))
        teen = string_file(get_abs_path('japanese/data/number/teen.tsv'))
        sign = string_file(get_abs_path('japanese/data/number/sign.tsv'))
        dot = string_file(get_abs_path('japanese/data/number/dot.tsv'))

        rmzero = delete('0') | delete('０')
        rmpunct = delete(',').ques
        rmspace = delete(' ').ques

        # 0-9
        digits = zero | digit
        en_digits = zero | en_digit
        self.digits = digits
        # 10-99
        ten = teen + insert('十') + (digit | rmzero)
        # 100-999
        hundred = (teen + insert('百') + (ten | (rmzero + digit) | rmzero**2))
        # 1000-9999
        thousand = (teen + insert('千') + rmpunct + (hundred
                                                    | (rmzero + ten)
                                                    | (rmzero**2 + digit)
                                                    | rmzero**3))
        self.thousand = thousand
        # 10000-99999999
        ten_thousand = ((thousand | hundred | ten | digit) + insert('万') +
                        (thousand
                         | (rmzero + rmpunct + hundred)
                         | (rmzero + rmpunct + rmzero + ten)
                         | (rmzero + rmpunct + rmzero + rmzero + digit)
                         | rmzero**4))
        # 0-99999999
        number = digits | ten | hundred | thousand | ten_thousand
        self.thousands = digits | ten | hundred | thousand
        # ±0.0 - ±99999999.99999999
        number = sign.ques + number + (dot + en_digits.plus).ques
        self.number = number

        # % like -27.00%
        percent = number + delete('%') + insert('パーセント')
        # ip like 127.0.0.1
        ip_dot = cross('.', 'ドット')
        ip = en_digits.plus + (ip_dot + en_digits.plus)**3
        # phone like 0xx-xxxx-xxxx
        country_code = (cross('+81', 'プラス八一') + cross('-', 'の').ques + rmspace)
        phone = (zero + en_digit + zero.ques + cross('-', 'の') + en_digits**3 +
                 en_digits.ques + cross('-', 'の').ques + en_digits**4)
        phone = country_code.ques + phone
        # others like 342388491
        others = en_digits**8 + en_digits.plus

        number = number | percent | ip | phone | others
        tagger = insert('value: "') + number.optimize() + insert('"')
        self.tagger = self.add_tokens(tagger)
