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


class Cardinal(Processor):
    """Cardinals up to 9999, e.g. ``9837`` -> ``تسعة اَلاف وثمان مئة وسبعة وثلاثون``.

    The tagger keeps the written digits (and an optional ``-``); the
    verbalizer spells them out. Every place value consumes exactly one digit.
    """

    def __init__(self):
        super().__init__("cardinal", token_orders={})
        self.build_graphs()
        self.build_tagger()
        self.build_verbalizer()

    def build_graphs(self):
        digit = pynini.string_file(get_abs_path("arabic/data/number/digit.tsv"))
        digit_100 = pynini.string_file(get_abs_path("arabic/data/number/digit_100.tsv"))
        digit_1000 = pynini.string_file(get_abs_path("arabic/data/number/digit_1000.tsv"))
        teens = pynini.string_file(get_abs_path("arabic/data/number/teens.tsv"))
        # tens.tsv also lists 10 -> مئة, which would let "100" pose as two digits.
        tens = self.DIGIT @ pynini.string_file(get_abs_path("arabic/data/number/tens.tsv"))
        flops = pynini.string_file(get_abs_path("arabic/data/number/flops.tsv"))
        conjunction = insert(" و")

        # 21..99 are read ones first: 34 -> أربعة وثلاثون.
        compound = flops @ (digit + conjunction + tens)
        two_digits = teens | tens + delete("0") | compound
        self.below_hundred = (digit | two_digits).optimize()

        def tail(width, lower):
            """Remainder after a leading place value of ``width`` digits."""
            # NeMo's small penalty on round hundreds/thousands.
            rest = delete("0" * width, weight=0.001)
            for zeros, graph in enumerate(lower):
                rest |= delete("0" * zeros) + conjunction + graph
            return rest

        hundred_tail = tail(2, [two_digits, digit])
        hundreds = pynini.union(
            pynini.cross("1", "مئة"),
            pynini.cross("2", "مئتان"),
            digit_100 + insert(" مئة"),
        ) + hundred_tail
        self.hundreds = hundreds.optimize()

        thousand_tail = tail(3, [self.hundreds, two_digits, digit])
        thousands = pynini.union(
            pynini.cross("1", "ألف"),
            pynini.cross("2", "ألفان"),
            digit_1000 + insert(" اَلاف"),
        ) + thousand_tail

        self.graph = (self.below_hundred | self.hundreds | thousands).optimize()
        self.graph_with_leading_zeros = (delete("0").star + self.graph).optimize()
        self.sign = pynini.cross("-", "سالب ")

    def build_tagger(self):
        negative = self.tag_field("negative", self.sign) + insert(" ")
        integer = self.tag_field("integer", self.graph_with_leading_zeros)
        self.tagger = self.add_tokens(negative.ques + integer)

    def build_verbalizer(self):
        negative = self.verbalize_field("negative", self.sign) + delete(" ")
        integer = self.verbalize_field("integer", self.graph_with_leading_zeros)
        self.verbalizer = self.delete_tokens(negative.ques + integer)
