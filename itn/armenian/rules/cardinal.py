# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright 2015 and onwards Google, Inc.
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

from itn.armenian.rules.utils import preferred_map
from tn.processor import Processor


class Cardinal(Processor):
    """Spoken cardinals, e.g. ``իննսունյոթ`` -> ``97``.

    The tagger keeps the spoken words; the verbalizer assembles the digits in
    three-digit groups up to trillions.
    """

    def __init__(self):
        super().__init__(name="cardinal", ordertype="itn", token_orders={})
        self.build_graphs()
        self.tagger = self.add_tokens(self.tag_field("integer", self.graph))
        self.verbalizer = self.delete_tokens(self.verbalize_field("integer", self.graph))

    def build_graphs(self):
        delete_space = delete(self.SPACE).star
        # Case endings such as ``երեքին`` or ``քսաներեքն`` keep the number.
        ending = (delete("ն") | delete("ի") | delete("ին")).ques
        digit = preferred_map("armenian/data/numbers/digit.tsv") + ending
        digits_no_one = preferred_map("armenian/data/numbers/digits_no_one.tsv")
        ties = preferred_map("armenian/data/numbers/ties.tsv") + ending
        two_digits = (ties | insert("0")) + (digit | insert("0"))

        hundred = delete("հարյուր")
        first_digit = pynini.cross("հարյուր", "1") | digits_no_one + delete_space + hundred
        hundreds = (first_digit + delete_space | add_weight(insert("0"), 0.1)) + delete_space + two_digits
        hundreds = hundreds.optimize()
        nonzero = self.DIGIT.star + (self.DIGIT - "0") + self.DIGIT.star
        strip_zeros = delete("0").star + (self.DIGIT - "0") + self.DIGIT.star

        thousands = pynini.cross("հազար", "1") | hundreds + delete_space + delete("հազար")
        thousands = (thousands | add_weight(insert("000"), 1e-9)) + delete_space + hundreds
        millions = hundreds + delete_space + delete("միլիոն") | add_weight(insert("000"), 0.1)
        millions = millions + delete_space + thousands
        billions = hundreds + delete_space + delete("միլիարդ") + delete_space | add_weight(insert("000"), 0.1)
        billions = billions + millions
        trillions = hundreds + delete_space + delete("տրիլիոն") + delete_space | add_weight(insert("000"), 0.1)
        trillions = trillions + billions

        # Spaces only separate words inside a field, never surround it.
        trimmed = self.NOT_SPACE | self.NOT_SPACE + self.VSIGMA + self.NOT_SPACE
        trimmed = trimmed.optimize()
        graph = trillions @ strip_zeros | pynini.cross("զրո", "0")
        self.graph = (trimmed @ graph).optimize()
        self.hundreds = (trimmed @ (hundreds @ nonzero) @ strip_zeros).optimize()
