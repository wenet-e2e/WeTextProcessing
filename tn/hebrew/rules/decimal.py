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
from pynini.lib.pynutil import delete, insert

from tn.hebrew.rules import numbers
from tn.hebrew.rules.utils import DIGIT, NONZERO, SPACE, first_field
from tn.processor import Processor

# Halves, quarters and tenths of 1-9 are read as fractions.
FRACTIONS = {"5": "חצי", "25": "רבע", "75": "שלושת רבעי", "1": "עשירית"}


class Decimal(Processor):
    """Written decimals, e.g. ``4.85 מיליארד`` -> ``ארבע נקודה שמונים וחמש מיליארד``.

    The tagger keeps the integer digits, the digits after the point and an
    optional quantity word. ``1-9`` with ``.5``/``.25``/``.75``/``.1`` read
    as fractions (``שבעים ותשע וחצי``); otherwise the point is ``נקודה`` and a
    two-digit fraction without zeros is read as a number.
    """

    def __init__(self, cardinal):
        super().__init__(name="decimal")
        sep = delete(SPACE.plus) + insert(" ")
        integer = cardinal.number
        fractional = DIGIT.plus
        quantity = pynini.union("מיליון", "מיליארד", "אלף")
        self.number = ((self.tag_field("integer_part", integer) + insert(" ")).ques + delete(".")
                       + self.tag_field("fractional_part", fractional))
        self.number = self.number.optimize()
        quantified = (self.number + sep + self.tag_field("quantity", quantity)
                      | self.tag_field("integer_part", integer) + sep
                      + self.tag_field("quantity", pynini.union("מיליון", "מיליארד")))
        self.tagger = self.add_tokens(cardinal.prefix.ques + cardinal.negative.ques + (self.number | quantified))

        small = pynini.union(*[str(number) for number in range(1, 10)]).optimize()
        special = pynini.union(*FRACTIONS).optimize()
        as_fraction = pynini.string_map([(digits, " ו" + word) for digits, word in FRACTIONS.items()])
        whole = (pynini.cross("1", "אחד") | (cardinal.digits - "1") @ integer) + insert(" ")
        large = (cardinal.digits - small) @ whole
        two_digits = NONZERO + NONZERO
        digits = (two_digits @ numbers.cardinal_graph()
                  | (pynini.project(fractional, "input") - two_digits) @ numbers.digit_by_digit())
        point = insert("נקודה ") + digits
        fraction = self.verbalize_field("fractional_part", as_fraction)
        number = (first_field(self, "integer_part", small @ integer) + fraction
                  | first_field(self, "integer_part", whole).ques
                  + self.verbalize_field("fractional_part", (pynini.project(fractional, "input") - special) @ point)
                  | first_field(self, "integer_part", large) + self.verbalize_field("fractional_part", special @ point)
                  | self.verbalize_field("fractional_part", special @ point))
        self.verbalize_number = number.optimize()
        counted = (pynini.cross("1", "") | pynini.cross("2", "שני ")
                   | (cardinal.digits - "1" - "2") @ integer + insert(" "))
        quantity_field = self.verbalize_field("quantity", quantity)
        number = (self.verbalize_number + insert(" ") + delete(" ") + quantity_field
                  | first_field(self, "integer_part", counted) + quantity_field)
        self.verbalizer = self.delete_tokens(
            cardinal.verbalize_prefix.ques + cardinal.verbalize_negative.ques + (self.verbalize_number | number))
