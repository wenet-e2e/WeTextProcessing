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

    def __init__(self):
        super().__init__(name='cardinal')
        self.thousand = None  # used for year of date
        self.positive_integer = None  # used for sport
        self.number = None
        self.digits = None
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        zero = string_file(get_abs_path('japanese/data/number/zero.tsv'))
        en_zero = cross('0', 'ゼロ')
        # 1-9
        digit = string_file(get_abs_path('japanese/data/number/digit.tsv'))
        teen = string_file(get_abs_path('japanese/data/number/teen.tsv'))
        sign = string_file(get_abs_path('japanese/data/number/sign.tsv'))
        dot = string_file(get_abs_path('japanese/data/number/dot.tsv'))

        rmzero = delete('0') | delete('０')
        rmpunct = delete(',').ques
        rmspace = delete(' ').ques

        # 0-9
        digits = zero | digit
        self.digits = digits
        # 10-99
        ten = teen + insert('十') + (digit | rmzero)

        # 100-999
        hundred = (teen + insert('百') + (ten | (rmzero + digit) | rmzero**2))
        # hundred prefix containing "," like "1,11" in "1,115,000"
        hundred_prefix = (teen + insert('百') + rmpunct +
                          (ten | (rmzero + digit) | rmzero**2))

        # 1000-9999
        thousand = (teen + insert('千') + rmpunct + (hundred
                                                    | (rmzero + ten)
                                                    | (rmzero**2 + digit)
                                                    | rmzero**3))
        # thousand prefix containing "," like "10,00" in "10,000,000"
        thousand_prefix = (digit + insert('千') +
                           (hundred_prefix
                            | (rmzero + rmpunct + ten)
                            | (rmzero + rmpunct + rmzero + digit)
                            | rmzero + rmpunct + rmzero**2))
        self.thousand = thousand

        # 10000-99999999  e.g. 1,115,000  10,000,000
        ten_thousand = ((thousand_prefix | hundred_prefix | ten | digit) +
                        insert('万') +
                        (thousand
                         | (rmzero + rmpunct + hundred)
                         | (rmzero + rmpunct + rmzero + ten)
                         | (rmzero + rmpunct + rmzero + rmzero + digit)
                         | rmzero + rmpunct + rmzero**3))
        # 100,000,000+
        hundred_millon = ((thousand_prefix | hundred_prefix | ten | digit) +
                          insert('億') + rmzero**2 + rmpunct + rmzero**2 +
                          (thousand
                           | (rmzero + rmpunct + hundred)
                           | (rmzero + rmpunct + rmzero + ten)
                           | (rmzero + rmpunct + rmzero + rmzero + digit)
                           | rmzero + rmpunct + rmzero**3))
        # 0-99999999
        number = digits | ten | hundred | thousand | ten_thousand | hundred_millon
        self.positive_integer = number
        # ±0.0 - ±99999999.99999999
        number = sign.ques + number + (dot + digits.plus).ques
        self.number = number

        # % like -27.00%
        percent = number + delete('%') + insert('パーセント')
        # ip like 127.0.0.1
        ip = digits.plus + (dot + digits.plus)**3
        # phone like 0xx-xxxx-xxxx
        country_code = (cross('+81', 'プラス八一') + cross('-', 'の').ques + rmspace)
        en_digits = en_zero | digit
        phone = (en_zero + digit + en_zero.ques + cross('-', 'の') +
                 en_digits**3 + en_digits.ques + cross('-', 'の').ques +
                 en_digits**4)
        phone = country_code.ques + (
            phone | en_zero + digit + en_zero.ques + en_digits**8)
        # No. 番号
        ordinal = (accep('No.') | accep('番号') | accep('番号は')) + digits.plus
        room = digits.plus + accep('号室')
        # others like 342388491
        others = digits**8 + digits.plus

        number = number | percent | ip | phone | ordinal | room | others
        tagger = insert('value: "') + number.optimize() + insert('"')
        self.tagger = self.add_tokens(tagger)
