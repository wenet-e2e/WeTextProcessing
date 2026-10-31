# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from tn.processor import Processor
from tn.utils import get_abs_path

# The quantity is read first, whichever side the unit was written on.
TOKEN_ORDER = {
    "measure": [
        "negative", "integer", "integer_part", "fractional_part", "quantity", "numerator", "denominator", "vulgar",
        "units", "letters"
    ]
}


class Measure(Processor):
    """Quantities with units, e.g. ``%7`` / ``7%`` -> ``سبعة في المائة``.

    The quantity is a cardinal, decimal or fraction (flat raw fields reusing
    those rules' graphs). ``units`` holds a unit from measurements.tsv, with an
    optional ``/`` rate; ``letters`` holds the Latin letters of forms like
    ``5-kg`` or ``5x``, which are kept verbatim.
    """

    def __init__(self, cardinal, decimal, fraction):
        super().__init__("measure", token_orders=TOKEN_ORDER)
        self.cardinal = cardinal
        self.decimal = decimal
        self.fraction = fraction
        self.build_graphs()
        self.build_tagger()
        self.build_verbalizer()

    def build_graphs(self):
        unit = pynini.string_file(get_abs_path("arabic/data/measure/measurements.tsv"))
        per = pynini.cross("/", "في")
        rate = per + insert(" ") + unit | unit + insert(" ") + per
        self.units = (unit + (insert(" ") + rate).ques | rate).optimize()
        self.letters = self.ALPHA.plus.optimize()
        self.times = pynini.union("x", "X")

    def build_tagger(self):
        cardinal = self.cardinal
        sign = self.tag_field("negative", cardinal.sign) + insert(" ")
        integer = self.tag_field("integer", cardinal.graph_with_leading_zeros)
        decimal = sign.ques + self.decimal.tag_number()
        quantity = sign.ques + integer | decimal | self.fraction.tag_number()
        units = self.tag_field("units", self.units)
        space = delete(" ").ques
        graph = quantity + space + insert(" ") + units | units + space + insert(" ") + quantity
        # Latin letters joined by a dash or an "x" multiplier keep their spelling.
        plain = self.tag_field("integer", cardinal.graph)
        letters = self.tag_field("letters", self.letters)
        graph |= (plain | decimal) + delete("-") + insert(" ") + letters
        graph |= letters + delete("-") + insert(" ") + (plain | decimal)
        graph |= (plain | decimal) + insert(" ") + self.tag_field("letters", self.times)
        self.tagger = self.add_tokens(graph)

    def build_verbalizer(self):
        cardinal = self.cardinal
        sign = self.verbalize_field("negative", cardinal.sign) + delete(" ")
        integer = self.verbalize_field("integer", cardinal.graph_with_leading_zeros)
        quantity = sign.ques + (integer | self.decimal.verbalize_number()) | self.fraction.verbalize_number()
        units = self.verbalize_field("units", self.units, True) | self.verbalize_field("letters", self.letters, True)
        self.verbalizer = self.delete_tokens(quantity + insert(" ") + units)
