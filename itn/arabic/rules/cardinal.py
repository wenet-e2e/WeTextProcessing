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
from pynini.lib import byte
from pynini.lib.pynutil import delete, insert

from itn.utils import get_abs_path
from tn.processor import Processor


def spell_out():
    """Written digits (1..9999) -> Arabic number words, one digit per place value."""

    digit = pynini.string_file(get_abs_path("arabic/data/number/digit.tsv"))
    digit_100 = pynini.string_file(get_abs_path("arabic/data/number/digit_100.tsv"))
    digit_1000 = pynini.string_file(get_abs_path("arabic/data/number/digit_1000.tsv"))
    teens = pynini.string_file(get_abs_path("arabic/data/number/teens.tsv"))
    # tens.tsv also lists 10 -> مئة, which would let "100" pose as two digits.
    tens = byte.DIGIT @ pynini.string_file(get_abs_path("arabic/data/number/tens.tsv"))
    flops = pynini.string_file(get_abs_path("arabic/data/number/flops.tsv"))
    conjunction = insert(" و")
    two_digits = teens | tens + delete("0") | flops @ (digit + conjunction + tens)

    def tail(width, lower):
        # Round hundreds/thousands cost a little more, so a decimal like
        # "مئة وواحد وخمسون من مئة" keeps the longer integer part (101.50).
        rest = delete("0" * width, weight=0.001)
        for zeros, graph in enumerate(lower):
            rest |= delete("0" * zeros) + conjunction + graph
        return rest

    hundreds = pynini.union(pynini.cross("1", "مئة"), pynini.cross("2", "مئتان"), digit_100 + insert(" مئة"))
    hundreds = (hundreds + tail(2, [two_digits, digit])).optimize()
    thousands = pynini.union(pynini.cross("1", "ألف"), pynini.cross("2", "ألفان"), digit_1000 + insert(" اَلاف"))
    thousands += tail(3, [hundreds, two_digits, digit])
    return (digit | two_digits | hundreds | thousands).optimize()


class Cardinal(Processor):
    """Number words up to 9999, e.g. ``سالب تسعة وتسعون`` -> ``-99``.

    The tagger keeps the spoken words; the verbalizer writes the digits.
    """

    def __init__(self):
        super().__init__(name="cardinal", ordertype="itn")
        self.spoken = spell_out()
        self.graph = pynini.invert(self.spoken).optimize()
        self.negative = pynini.cross("سالب", "-")
        sign = self.tag_field("negative", self.negative) + delete(" ") + insert(" ")
        self.tagger = self.add_tokens(sign.ques + self.tag_field("integer", self.graph))
        sign_out = self.verbalize_field("negative", self.negative) + delete(" ")
        self.verbalizer = self.delete_tokens(sign_out.ques + self.verbalize_field("integer", self.graph))
