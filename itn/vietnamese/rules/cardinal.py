# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
from pynini.lib.pynutil import delete, insert

from itn.utils import get_abs_path
from tn.processor import Processor


WHITE_SPACE = pynini.union(" ", "\t", "\n", "\r", "\u00a0").optimize()
# NeMo's ``delete_space`` (any whitespace, possibly none) and ``delete_extra_space`` (at least one).
DELETE_SPACE = delete(WHITE_SPACE.star).optimize()
DELETE_EXTRA_SPACE = delete(WHITE_SPACE.plus).optimize()


def numbers_file(name):
    return pynini.string_file(get_abs_path("vietnamese/data/numbers/" + name + ".tsv")).optimize()


class Cardinal(Processor):
    """Spoken integers keep their words raw (``âm hai mươi lăm``) and become digits when verbalized."""

    def __init__(self):
        super().__init__(name="cardinal", ordertype="itn", token_orders={})
        ties = numbers_file("ties")
        teen = numbers_file("teen")
        zero_word = numbers_file("zero")
        digit = numbers_file("digit")
        self.digit, self.zero = digit, zero_word
        thousand_words = pynini.union("ngàn", "nghìn")
        hundred = delete("trăm")
        ten = delete("mươi")
        linh = pynini.cross(pynini.union("linh", "lẻ"), "0")
        self.one = pynini.cross("mốt", "1")
        self.four = pynini.cross("tư", "4")
        self.five = pynini.cross("lăm", "5")
        self.half = pynini.cross("rưỡi", "5")
        ds = DELETE_SPACE

        optional_ten = (ds + ten).ques
        # "năm" never ends a number after a scale word ("ba nghìn năm" is 3000 + "năm").
        self.last_digit = (
            pynini.difference(self.input_projection(digit), "năm") @ digit | self.one | self.four | self.five
        ).optimize()
        hundreds = (digit | zero_word) + ds + hundred + ds + pynini.union(
            teen,
            (self.half | self.four | self.one) + insert("0", weight=0.1),
            ties + optional_ten + (ds + self.last_digit | insert("0", weight=0.1)),
            linh + ds + (digit | self.four | self.five),
            insert("00", weight=0.1),
        )
        hundreds |= insert("0") + ds + pynini.union(
            teen,
            ties + optional_ten + ds + self.last_digit,
            ties + ds + ten + insert("0", weight=0.1),
            linh + ds + (digit | self.four | self.five),
        )
        hundreds = hundreds.optimize()
        component = (hundreds | insert("00", weight=0.1) + ds + digit).optimize()
        self.nonzero_component = (component @ (self.DIGIT.star + (self.DIGIT - "0") + self.DIGIT.star)).optimize()
        hundreds_zero = hundreds | insert("000", weight=0.1)

        def scale(words):
            return (self.nonzero_component + ds + delete(words) | insert("000", weight=0.1)).optimize()

        graph = pynini.union(
            scale(pynini.union("tỉ", "tỷ")) + ds + scale("triệu") + ds + scale(thousand_words) + ds + hundreds_zero,
            self.nonzero_component + ds + delete(thousand_words) + ds
            + ((self.last_digit | self.half) + insert("00", weight=0.1) | hundreds_zero),
            digit,
            zero_word,
        )
        self.strip_zeros = (delete("0").star + (self.DIGIT - "0") + self.DIGIT.star | "0").optimize()
        self.graph_no_exception = (graph @ self.strip_zeros).optimize()
        # Bare digits zero to nine stay words in running text.
        single = self.input_projection(digit | zero_word)
        self.graph = (
            pynini.difference(self.input_projection(self.graph_no_exception), single) @ self.graph_no_exception
        ).optimize()

        self.negative = pynini.cross(pynini.union("âm", "trừ"), "-")
        sign = self.tag_field("negative", self.negative) + pynini.accep(" ")
        self.tagger = self.add_tokens(sign.ques + self.tag_field("integer", self.graph))
        sign_v = self.verbalize_field("negative", self.negative) + delete(" ")
        self.verbalizer = self.delete_tokens(sign_v.ques + self.verbalize_field("integer", self.graph))
