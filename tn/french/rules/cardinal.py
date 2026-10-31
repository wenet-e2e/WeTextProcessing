# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pynini
from pynini.lib.pynutil import add_weight, delete, insert

from tn.processor import Processor
from tn.utils import get_abs_path


TOKEN_ORDER = {"cardinal": ["negative", "integer"]}


class Cardinal(Processor):
    """Tags written digits verbatim; spells them out in the verbalizer.

    ``23`` -> ``cardinal { integer: "23" }`` -> ``vingt-trois``. Digit groups
    may be separated by single spaces (``6 492 000``), up to the milliards.
    """

    def __init__(self):
        super().__init__("cardinal", token_orders=TOKEN_ORDER)
        self.build_graphs()
        self.build_tagger()
        self.build_verbalizer()

    def build_graphs(self):
        def load(name):
            return pynini.string_file(get_abs_path("french/data/numbers/{}.tsv".format(name)))

        space = insert(" ")
        zero = load("zero")
        digits = load("digit")
        digits_no_one = (self.DIGIT - "1") @ digits
        one = "1" @ digits

        ten = load("ten")
        teens = load("teens")
        teens_no_one = (pynini.accep("1") + (self.DIGIT - "1")) @ teens

        tens = load("tens_simple")
        ties_simple = tens + (delete("0") | insert("-") + digits_no_one | space + pynini.cross("1", "et un"))

        seventy = load("seventy")
        seventy_unit = insert("-") + ((insert("1") + self.DIGIT) @ teens_no_one)
        seventies = seventy + (pynini.cross("0", "-dix") | seventy_unit | space + pynini.cross("1", "et onze"))

        eighties = load("eighty") + (pynini.cross("0", "s") | insert("-") + digits)

        nineties_unit = insert("-") + ((insert("1") + self.DIGIT) @ teens)
        nineties = load("ninety") + (pynini.cross("0", "-dix") | nineties_unit)

        self.double_digits = (ten | teens | ties_simple | seventies | eighties | nineties).optimize()
        self.one_to_all_tens = (digits | self.double_digits).optimize()

        hundreds_parts = delete("0") + space + digits | space + self.double_digits
        one_hundreds = pynini.cross("1", "cent") + (delete("00") | hundreds_parts)
        multiple_hundreds = digits_no_one + space + insert("cent") + (pynini.cross("00", "s") | hundreds_parts)
        self.hundreds = (one_hundreds | multiple_hundreds).optimize()

        separator = delete(" ").ques
        one_thousand = pynini.cross("1", "mille") + separator
        other_thousands = (digits_no_one | self.double_digits | self.hundreds) + space + insert("mille") + separator
        thousands = (
            (one_thousand | other_thousands) + delete("000")
            | one_thousand + delete("00") + space + (pynini.cross("1", "et un") | digits_no_one)
            | other_thousands + delete("00") + space + digits
            | (one_thousand | other_thousands) + delete("0") + space + self.double_digits
            | (one_thousand | other_thousands) + space + self.hundreds
        )

        to_hundreds_no_one = digits_no_one | self.double_digits | self.hundreds
        to_thousands_no_one = to_hundreds_no_one | thousands
        lower = space + (to_thousands_no_one | pynini.cross("1", "et un"))
        millions_components = pynini.closure(separator + pynini.closure(self.DIGIT, 3), 2)
        millions = (one + space + insert("million") | to_hundreds_no_one + space + insert("millions")) + (
            millions_components @ (pynini.closure(delete("0"), 0, 6) + lower.ques))

        to_millions_no_one = to_thousands_no_one | millions
        lower = space + (to_millions_no_one | pynini.cross("1", "et un"))
        billions_components = pynini.closure(separator + pynini.closure(self.DIGIT, 3), 3)
        billions = (one + space + insert("milliard") | to_hundreds_no_one + space + insert("milliards")) + (
            billions_components @ (pynini.closure(delete("0"), 0, 9) + lower.ques))

        # Masculine reading first; the feminine "-une" ending is a weighted alternative.
        self.graph = (zero | self.one_to_all_tens | self.hundreds | thousands | millions | billions).optimize()
        self.digits = (zero | digits).optimize()
        feminine = self.graph @ (self.VSIGMA + pynini.cross("un", "une"))
        self.gendered = (self.graph | add_weight(feminine, 0.01)).optimize()
        self.sign = pynini.cross("-", "moins")

    def build_tagger(self):
        sign = self.tag_field("negative", self.sign) + insert(" ")
        self.tagger = self.add_tokens(sign.ques + self.tag_field("integer", self.graph))

    def build_verbalizer(self):
        sign = self.verbalize_field("negative", self.sign) + pynini.accep(" ")
        self.verbalizer = self.delete_tokens(sign.ques + self.verbalize_field("integer", self.gendered))
