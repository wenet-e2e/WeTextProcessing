# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from itn.french.rules.cardinal import load
from itn.french.rules.decimal import WHITE_SPACE
from tn.processor import Processor


class Telephone(Processor):
    """Ten-digit French numbers read in pairs or digit by digit.

    ``zéro un quatre-vingt-deux zéro deux vingt-deux cinquante`` -> ``01 82 02 22 50``.
    A missing leading zero is restored, and ``double neuf`` reads ``99``.
    """

    def __init__(self):
        super().__init__("telephone", ordertype="itn", token_orders={})
        delete_space = delete(WHITE_SPACE.star)
        space = insert(" ")
        digit = load("numbers/digit")
        zero = load("numbers/zero")
        ties = load("numbers/ties")
        double_digits = pynini.union(
            load("numbers/teen"), load("numbers/ties_unique"), ties + insert("0"), ties + delete("-").ques + digit)

        single_digits = digit | zero
        words = (digit.optimize() | zero).invert()
        # "double neuf" -> "neuf neuf" -> "99"
        doubled = []
        for i in range(10):
            word = pynini.project(str(i) @ words, "output")
            doubled.append(pynini.cross(pynini.accep("double ") + word, word + " " + word))
        doubled = pynini.union(*doubled) @ (single_digits + delete(" ") + single_digits)

        first_pair = zero + delete_space + digit | insert("0") + digit | doubled
        first_pair += delete_space + space
        pair = single_digits + delete_space + single_digits | doubled
        all_digits = first_pair + pynini.closure(pair + delete_space + space, 3, 3) + pair
        pair = double_digits | pair
        digits_and_ties = first_pair + pynini.closure(pair + delete_space + space, 3, 3) + pair
        graph = (all_digits | digits_and_ties).optimize()

        self.tagger = self.add_tokens(self.tag_field("number_part", graph))
        self.verbalizer = self.delete_tokens(self.verbalize_field("number_part", graph))
