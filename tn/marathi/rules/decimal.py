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

from tn.marathi.rules.cardinal import MINUS
from tn.marathi.rules.utils import MR_DIGIT, canonical
from tn.processor import Processor

POINT = "पूर्णांक"
QUANTITIES = ("हजार", "लाख", "कोटी", "अब्ज")


class Decimal(Processor):
    """Written decimals, e.g. ``-१२.१५ कोटी`` -> ``उणे बारा पूर्णांक एक पाच कोटी``.

    The tagger keeps the written sign, integer digits, fractional digits and
    quantity word and drops the decimal point; the verbalizer reads the
    integer as a cardinal and the fraction digit by digit.
    """

    def __init__(self, cardinal):
        super().__init__("decimal", token_orders={})
        self.integer = cardinal.integer
        self.fractional = canonical(MR_DIGIT.plus @ cardinal.digits)
        self.quantity = pynini.union(*QUANTITIES).optimize()
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        negative = self.tag_field("negative", pynini.accep("-")) + insert(" ")
        integer = self.tag_field("integer_part", self.integer)
        fractional = self.tag_field("fractional_part", self.fractional)
        decimal = (integer + insert(" ")).ques + delete(".") + fractional
        quantity = delete(self.SPACE.plus) + insert(" ") + self.tag_field("quantity", self.quantity)
        graph = decimal + quantity.ques | integer + quantity
        self.tagger = self.add_tokens(negative.ques + graph)

    def build_verbalizer(self):
        sign = self.verbalize_field("negative", pynini.cross("-", MINUS + " ")) + delete(" ")
        integer = self.verbalize_field("integer_part", self.integer)
        fractional = self.verbalize_field("fractional_part", self.fractional, leading_space=True)
        decimal = integer + insert(" " + POINT + " ") + fractional
        decimal |= insert(POINT + " ") + self.verbalize_field("fractional_part", self.fractional)
        quantity = insert(" ") + self.verbalize_field("quantity", self.quantity, leading_space=True)
        graph = decimal + quantity.ques | integer + quantity
        self.verbalizer = self.delete_tokens(sign.ques + graph)
