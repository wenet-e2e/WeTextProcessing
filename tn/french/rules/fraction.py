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
from pynini.lib.pynutil import add_weight, delete, insert

from tn.processor import Processor
from tn.utils import get_abs_path


TOKEN_ORDER = {"fraction": ["negative", "integer_part", "numerator", "denominator"]}


class Fraction(Processor):
    """``-4 1/3`` keeps every number verbatim -> ``moins quatre et un tiers``."""

    def __init__(self, cardinal, ordinal):
        super().__init__("fraction", token_orders=TOKEN_ORDER)
        digits = self.DIGIT.star
        singular = (pynini.accep("1") @ cardinal.graph).optimize()
        plural = ((digits - "1") @ cardinal.graph).optimize()

        irregular = pynini.string_file(get_abs_path("french/data/fractions/irregular_denominators.tsv"))

        def denominator(suffix):
            graph = ordinal.semantic(digits @ cardinal.graph, insert(suffix))
            return pynini.union(add_weight(graph @ irregular, -0.01), graph).optimize()

        singular_denominator = denominator("ième")
        plural_denominator = denominator("ièmes")
        denominators = singular_denominator | plural_denominator

        sign = self.tag_field("negative", cardinal.sign) + insert(" ")
        integer = self.tag_field("integer_part", cardinal.graph) + delete(" ") + insert(" ")
        numerator = self.tag_field("numerator", singular | plural) + delete("/") + insert(" ")
        fraction = numerator + self.tag_field("denominator", denominators)
        self.tagger = self.add_tokens(sign.ques + integer.ques + fraction)

        sign = self.verbalize_field("negative", cardinal.sign) + pynini.accep(" ")
        integer = self.verbalize_field("integer_part", cardinal.graph) + delete(" ") + insert(" et ")
        fraction = self.verbalize_field("numerator", singular) + pynini.accep(" ")
        fraction += self.verbalize_field("denominator", singular_denominator)
        fraction |= self.verbalize_field("numerator", plural) + pynini.accep(" ") + self.verbalize_field(
            "denominator", plural_denominator)
        graph = integer.ques + fraction
        half = self.VSIGMA + pynini.cross("et un demi", "et demi")
        graph = pynini.union(add_weight(graph @ half, -0.01), graph)
        self.verbalizer = self.delete_tokens(sign.ques + graph)
