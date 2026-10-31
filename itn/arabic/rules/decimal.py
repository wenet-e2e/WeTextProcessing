# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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


class Decimal(Processor):
    """Decimals, e.g. ``ستة وثمانية وثمانون من مئة`` -> ``6.88``.

    The spoken denominator (عشرة, مئة, ألف, عشرة آلاف) fixes the number of
    fractional digits. The tagger keeps the words, dropping only the ``و``
    that joins the two parts.
    """

    def __init__(self, cardinal):
        super().__init__(name="decimal", ordertype="itn")
        self.cardinal = cardinal
        padded = delete("0").star + cardinal.spoken
        denominators = ["عشرة", "مئة", "ألف", "عشرة آلاف"]
        fractional = pynini.union(*[
            self.DIGIT**width @ padded + insert(" من " + denominator)
            for width, denominator in enumerate(denominators, 1)
        ])
        self.fractional = pynini.invert(fractional).optimize()
        self.quantity = pynini.string_file(get_abs_path("arabic/data/number/quantities.tsv")).optimize()
        self.negative = cardinal.negative
        sign = self.tag_field("negative", self.negative) + self.DELETE_EXTRA_SPACE
        self.tagger = self.add_tokens(sign.ques + self.tag_number())
        sign_out = self.verbalize_field("negative", self.negative) + delete(" ")
        self.verbalizer = self.delete_tokens(sign_out.ques + self.verbalize_number())

    def tag_number(self):
        """Raw decimal fields without the sign, shared with measure."""

        tagger = self.tag_field("integer_part", self.cardinal.graph) + pynini.accep(" ") + delete("و")
        tagger += self.tag_field("fractional_part", self.fractional)
        return tagger + (pynini.accep(" ") + self.tag_field("quantity", self.quantity)).ques

    def verbalize_number(self):
        verbalizer = self.verbalize_field("integer_part", self.cardinal.graph) + insert(".")
        verbalizer += self.verbalize_field("fractional_part", self.fractional, True)
        return verbalizer + (insert(" ") + self.verbalize_field("quantity", self.quantity, True)).ques
