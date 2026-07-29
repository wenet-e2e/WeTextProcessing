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

from pynini import closure, cross, difference, string_file, union
from pynini.lib.pynutil import add_weight, delete, insert

from itn.english.rules.cardinal import Cardinal
from tn.processor import Processor
from itn.utils import get_abs_path


class Telephone(Processor):

    def __init__(self, cardinal=None):
        super().__init__(name="telephone", ordertype="itn")
        self.cardinal = cardinal or Cardinal()
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        ds = delete(" ")
        digit = string_file(get_abs_path("english/data/numbers/digit.tsv"))
        zero = string_file(get_abs_path("english/data/numbers/zero.tsv"))
        single = digit | zero | cross("o", "0") | cross("oh", "0")

        # "double X" => XX
        double = union(*[
            cross(f"double {w}", f"{d}{d}")
            for w, d in [("one", "1"), ("two", "2"), ("three",
                                                      "3"), ("four",
                                                             "4"), ("five",
                                                                    "5"), ("six",
                                                                           "6"), ("seven",
                                                                                  "7"), ("eight",
                                                                                         "8"), ("nine",
                                                                                                "9"), ("zero",
                                                                                                       "0"), ("oh",
                                                                                                              "0"), ("o", "0")]
        ])

        # "triple X" => XXX
        triple = union(*[
            cross(f"triple {w}", f"{d}{d}{d}")
            for w, d in [("one", "1"), ("two", "2"), ("three",
                                                      "3"), ("four",
                                                             "4"), ("five",
                                                                    "5"), ("six",
                                                                           "6"), ("seven",
                                                                                  "7"), ("eight",
                                                                                         "8"), ("nine",
                                                                                                "9"), ("zero",
                                                                                                       "0"), ("oh",
                                                                                                              "0"), ("o", "0")]
        ])

        # two-digit cardinal: twenty three => 23 (uses graph_two_digit for proper space handling)
        two_digit = self.cardinal.graph_two_digit

        # a token is 1, 2, or 3 digits
        token = single | double | triple | add_weight(two_digit, 0.002)

        # sequence of tokens separated by spaces
        seq = token + closure(ds + token)

        # phone: XXX-XXX-XXXX
        phone = seq @ (self.DIGIT**3 + insert("-") + self.DIGIT**3 + insert("-") + self.DIGIT**4)

        # country code
        country_digits = (single
                          | add_weight(single + ds + single, -0.001)
                          | add_weight(single + ds + single + ds + single, -0.002)
                          | add_weight(two_digit, 0.002))
        self.country_code = closure(cross("plus ", "+"), 0, 1) + country_digits
        country_code = self.tag_field("country_code", self.country_code)
        optional_cc = closure(country_code + ds + insert(" "), 0, 1)

        def number_field(graph, kind):
            return self.tag_field("number_part", graph) + insert(f' kind: "{kind}"')

        self.number_parts = {"phone": phone}
        graph = optional_cc + number_field(phone, "phone")

        # SSN: XXX-XX-XXXX
        ssn = seq @ (self.DIGIT**3 + insert("-") + self.DIGIT**2 + insert("-") + self.DIGIT**4)
        self.number_parts["ssn"] = ssn
        graph |= number_field(ssn, "ssn")

        # IP: X.X.X.X
        ip_token = (single + closure(ds + single, 0, 2)
                    | double
                    | triple
                    | add_weight(two_digit, 0.002)
                    | single + ds + two_digit
                    | two_digit + ds + single)
        ip = ip_token + (cross(" dot ", ".") + ip_token)**3
        self.number_parts["ip"] = ip
        graph |= add_weight(number_field(ip, "ip"), -0.001)

        # credit card: 4-4-4-4 (16), 4-6-4 (14), 4-6-5 (15)
        space = insert(" ")
        D = self.DIGIT
        cc_format = (D**4 + space + D**4 + space + D**4 + space + D**4
                     | D**4 + space + D**6 + space + D**4
                     | D**4 + space + D**6 + space + D**5)
        cc = seq @ cc_format
        self.number_parts["credit_card"] = cc
        graph |= optional_cc + number_field(cc, "credit_card")

        # serial: mixed alpha+digits, at least one digit, length >= 3
        # Exclude "a" as first char to avoid "a thirty six" -> "a36"
        not_a = difference(self.ALPHA, union("a", "A"))
        serial_digit = single | add_weight(two_digit, -0.002)
        serial_char = serial_digit | self.ALPHA
        seq1 = (not_a | serial_digit) + closure(ds + serial_char, 2)
        seq1 |= serial_char + closure(ds + (single | self.ALPHA), 2)
        seq2 = self.ALPHA + closure(ds + self.ALPHA, 1) + closure(ds + two_digit, 1)
        seq2 |= not_a + closure(ds + two_digit, 1)
        seq2 |= two_digit + closure(ds + two_digit, 1) + closure(ds + self.ALPHA, 1)
        serial = (seq1 | seq2) @ (closure(self.ALPHA | D) + D + closure(self.ALPHA | D))
        serial = add_weight(serial, 2.0)
        self.number_parts["serial"] = serial
        graph |= number_field(serial, "serial")

        self.tagger = self.add_tokens(graph)

    def build_verbalizer(self):
        cc = self.verbalize_field("country_code", self.country_code)
        graphs = []
        for kind, number_part in self.number_parts.items():
            num = self.verbalize_field("number_part", number_part, leading_space=True)
            num_only = self.verbalize_field("number_part", number_part)
            marker = delete(f' kind: "{kind}"')
            graphs.append(cc + self.DELETE_SPACE + insert(" ") + num + marker)
            graphs.append(num_only + marker)
        graph = union(*graphs)
        self.verbalizer = self.delete_tokens(graph)
