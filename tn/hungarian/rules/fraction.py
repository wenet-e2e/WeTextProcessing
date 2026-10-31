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
from pynini.lib.pynutil import delete, insert

from tn.hungarian.rules.utils import data_map
from tn.processor import Processor

TOKEN_ORDER = {}


class Fraction(Processor):
    """Fractions, e.g. ``2/3`` -> ``kettő harmad``, ``1 1/2`` -> ``egy és egy fél``.

    Raw fields: ``integer_part``, ``numerator``, ``denominator`` or a
    vulgar fraction ``symbol`` such as ``⅔``.
    """

    def __init__(self, cardinal, ordinal):
        super().__init__("fraction", token_orders=TOKEN_ORDER)
        numerator = cardinal.graph
        denominator = ordinal.fractional
        self.symbol = (data_map("fractions/fraction_symbols.tsv")
                       @ (numerator + pynini.cross("/", " ") + denominator)).optimize()

        integer_t = self.tag_field("integer_part", cardinal.graph) + pynini.accep(" ")
        fraction_t = (
            self.tag_field("numerator", numerator) + delete(pynini.union("/", " / ")) + insert(" ")
            + self.tag_field("denominator", denominator)
            | self.tag_field("symbol", self.symbol)
        )
        self.unsigned_tagger = (integer_t.ques + fraction_t).optimize()
        self.tagger = self.add_tokens(cardinal.sign_tagger + self.unsigned_tagger)

        integer_v = self.verbalize_field("integer_part", cardinal.graph) + pynini.cross(" ", " és ")
        fraction_v = (
            self.verbalize_field("numerator", numerator) + pynini.accep(" ")
            + self.verbalize_field("denominator", denominator)
            | self.verbalize_field("symbol", self.symbol)
        )
        self.unsigned_verbalizer = (integer_v.ques + fraction_v).optimize()
        self.verbalizer = self.delete_tokens(cardinal.sign_verbalizer + self.unsigned_verbalizer)
