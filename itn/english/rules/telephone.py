# Copyright (c) 2026 Zhendong Peng (pzd17@tsinghua.org.cn)
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

from pynini import closure, cross, string_file, union
from pynini.lib.pynutil import add_weight, delete, insert

from itn.english.rules.cardinal import Cardinal
from tn.processor import Processor
from tn.utils import get_abs_path


class Telephone(Processor):

    def __init__(self, cardinal=None):
        super().__init__(name="telephone", ordertype="itn")
        self.cardinal = cardinal or Cardinal()
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        ds = delete(" ")
        digit = string_file(get_abs_path("../itn/english/data/numbers/digit.tsv"))
        zero = string_file(get_abs_path("../itn/english/data/numbers/zero.tsv"))
        single = digit | zero | cross("o", "0") | cross("oh", "0")

        # "double X" => XX
        double = union(*[cross(f"double {w}", f"{d}{d}")
                         for w, d in [("one","1"),("two","2"),("three","3"),("four","4"),
                                      ("five","5"),("six","6"),("seven","7"),("eight","8"),
                                      ("nine","9"),("zero","0"),("oh","0"),("o","0")]])

        # two-digit cardinal: twenty three => 23
        two_digit = self.cardinal.graph_no_exception @ (self.DIGIT + self.DIGIT)

        # a token is 1 or 2 digits
        token = single | double | add_weight(two_digit, 0.002)

        # sequence of tokens separated by spaces
        seq = token + closure(ds + token)

        # phone: XXX-XXX-XXXX
        phone = seq @ (
            self.DIGIT ** 3 + insert("-") + self.DIGIT ** 3 + insert("-") + self.DIGIT ** 4
        )

        # country code
        country_code = (
            insert('country_code: "')
            + closure(cross("plus ", "+"), 0, 1)
            + (closure(single + ds, 0, 2) + single | add_weight(two_digit, 0.002))
            + insert('"')
        )
        optional_cc = closure(country_code + ds + insert(" "), 0, 1)

        graph = optional_cc + insert('number_part: "') + phone + insert('"')

        # SSN: XXX-XX-XXXX
        ssn = seq @ (
            self.DIGIT ** 3 + insert("-") + self.DIGIT ** 2 + insert("-") + self.DIGIT ** 4
        )
        graph |= insert('number_part: "') + ssn + insert('"')

        # IP: X.X.X.X
        ip_token = single + closure(ds + single, 0, 2) | double | add_weight(two_digit, 0.002)
        ip = ip_token + (cross(" dot ", ".") + ip_token) ** 3
        graph |= insert('number_part: "') + add_weight(ip, -0.001) + insert('"')

        # credit card: XXXX XXXX XXXX XXXX or XXXX XXXXXX XXXXX
        cc = seq @ (
            self.DIGIT ** 4 + insert(" ") + self.DIGIT ** 4
            + insert(" ") + self.DIGIT ** 4 + insert(" ") + self.DIGIT ** 4
        )
        graph |= insert('number_part: "') + cc + insert('"')

        # serial: mixed alpha+digits, at least one digit, length >= 3
        serial_char = single | add_weight(two_digit, 0.002) | self.ALPHA
        serial = serial_char + closure(ds + serial_char, 2)
        serial = serial @ (closure(self.ALPHA | self.DIGIT) + self.DIGIT + closure(self.ALPHA | self.DIGIT))
        graph |= insert('number_part: "') + add_weight(serial, 2.0) + insert('"')

        self.tagger = self.add_tokens(graph)

    def build_verbalizer(self):
        cc = delete('country_code: "') + self.NOT_QUOTE.plus + delete('"')
        num = delete(' number_part: "') + self.NOT_QUOTE.plus + delete('"')
        num_only = delete('number_part: "') + self.NOT_QUOTE.plus + delete('"')
        graph = cc + self.DELETE_SPACE + insert(" ") + num | num_only
        self.verbalizer = self.delete_tokens(graph)
