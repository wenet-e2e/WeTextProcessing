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


TOKEN_ORDER = {"fraction": ["negative", "integer_part", "numerator", "denominator"]}


class Fraction(Processor):
    """``un et deux tiers`` -> ``fraction { integer_part: "un" numerator: "deux" denominator: "tiers" }`` -> ``1 2/3``.

    A bare ``demi`` has no spoken numerator; its denominator field alone reads ``1/2``.
    """

    def __init__(self, cardinal):
        super().__init__("fraction", ordertype="itn", token_orders=TOKEN_ORDER)
        number = cardinal.graph_no_exception
        strip = delete("ième") + delete("s").ques | load("fractions")
        denominator = ((self.VSIGMA + strip) @ number).optimize()
        half = pynini.cross(pynini.accep("demi") + pynini.accep("e").ques + pynini.accep("s").ques, "1/2")
        delete_space = delete(WHITE_SPACE.star)
        self.sign = cardinal.sign

        integer = self.tag_field("integer_part", number) + delete_space + delete("et") + delete_space + insert(" ")
        fraction = self.tag_field("numerator", number) + delete_space + insert(" ")
        fraction = fraction + self.tag_field("denominator", denominator) | self.tag_field("denominator", half)
        self.number_tagger = integer.ques + fraction
        self.sign_tagger = self.tag_field("negative", self.sign) + delete(WHITE_SPACE.plus) + insert(" ")
        self.tagger = self.add_tokens(self.sign_tagger.ques + self.number_tagger)

        integer = self.verbalize_field("integer_part", number) + pynini.accep(" ")
        fraction = self.verbalize_field("numerator", number) + delete(" ") + insert("/")
        fraction = fraction + self.verbalize_field("denominator", denominator) | self.verbalize_field(
            "denominator", half)
        self.number_verbalizer = (integer.ques + fraction).optimize()
        self.sign_verbalizer = self.verbalize_field("negative", self.sign) + delete(" ")
        self.verbalizer = self.delete_tokens(self.sign_verbalizer.ques + self.number_verbalizer)
