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
from pynini.lib.pynutil import add_weight, delete, insert

from itn.utils import get_abs_path
from tn.processor import Processor


class Fraction(Processor):
    """Fractions, e.g. ``واحد و نصف`` -> ``1 1/2`` and ``تسعة أثلاث`` -> ``9/3``.

    A lone singular (``نصف``) or dual (``ثلثين``) fraction noun implies the
    numerator 1 or 2; plural nouns follow a numerator from 3 to 10; other
    fractions are spoken ``X على Y``. The tagger keeps the words, dropping
    only the ``و`` and ``على`` joiners.
    """

    def __init__(self, cardinal):
        super().__init__(name="fraction", ordertype="itn")
        self.cardinal = cardinal
        number = cardinal.graph
        singular = pynini.string_file(get_abs_path("arabic/data/number/fraction_singular.tsv"))
        dual = pynini.string_file(get_abs_path("arabic/data/number/fraction_dual.tsv"))
        plural = pynini.string_file(get_abs_path("arabic/data/number/fraction_plural.tsv"))
        three_to_ten = pynini.union(*[str(n) for n in range(3, 11)])
        self.number = number
        self.singular = pynini.invert(singular).optimize()
        self.dual = pynini.invert(dual).optimize()
        self.plural = pynini.invert(plural).optimize()
        self.three_to_ten = (number @ three_to_ten).optimize()
        self.negative = cardinal.negative
        sign = self.tag_field("negative", self.negative) + self.DELETE_SPACE + insert(" ")
        self.tagger = self.add_tokens(sign.ques + self.tag_number())
        sign_out = self.verbalize_field("negative", self.negative) + delete(" ")
        self.verbalizer = self.delete_tokens(sign_out.ques + self.verbalize_number())

    def tag_number(self):
        """Raw fraction fields without the sign, shared with measure."""

        optional_space = self.DELETE_ZERO_OR_ONE_SPACE
        integer = self.tag_field("integer_part", self.number) + optional_space + delete("و") + optional_space
        integer += insert(" ")
        lone = self.tag_field("denominator", self.singular | self.dual)
        plural = self.tag_field("numerator", self.three_to_ten) + optional_space + insert(" ")
        plural += self.tag_field("denominator", self.plural)
        ratio = self.tag_field("numerator", self.number) + optional_space + delete("على") + optional_space
        ratio += insert(" ") + self.tag_field("denominator", self.number)
        return integer.ques + (lone | plural | add_weight(ratio, 0.001))

    def verbalize_number(self):
        integer = self.verbalize_field("integer_part", self.number) + pynini.accep(" ")
        lone = insert("1/") + self.verbalize_field("denominator", self.singular)
        lone |= insert("2/") + self.verbalize_field("denominator", self.dual)
        plural = self.verbalize_field("numerator", self.three_to_ten) + insert("/")
        plural += self.verbalize_field("denominator", self.plural, True)
        ratio = self.verbalize_field("numerator", self.number) + insert("/")
        ratio += self.verbalize_field("denominator", self.number, True)
        return integer.ques + (lone | plural | ratio)
