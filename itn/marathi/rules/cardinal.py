# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

from itn.marathi.rules.utils import DIGIT, NON_ZERO, load_pairs, spoken_map
from tn.processor import Processor

MINUS = "उणे"
# Indian place values above the hundreds, each spanning two digits.
SCALES = ("अब्ज", "कोटी", "लाख", "हजार")
# Skipping a place value costs a little, so ``चार लाख`` prefers the decimal
# quantity reading ``४ लाख`` over the cardinal ``४०००००``.
SKIP_WEIGHT = 0.1


class Cardinal(Processor):
    """Spoken cardinals, e.g. ``उणे सातशे बावीस`` -> ``-७२२``.

    The tagger keeps the spoken words (and the ``उणे`` sign word); the
    verbalizer assembles Devanagari digits in Indian grouping (lakh, crore,
    abja) and also reads ``एकतीसशे`` style hundreds counts.
    """

    def __init__(self):
        super().__init__(name="cardinal", ordertype="itn", token_orders={})
        self.build_graphs()
        negative = self.tag_field("negative", pynini.accep(MINUS)) + delete(self.SPACE.plus) + insert(" ")
        self.tagger = self.add_tokens(negative.ques + self.tag_field("integer", self.graph))
        sign = self.verbalize_field("negative", pynini.cross(MINUS, "-")) + delete(" ")
        self.verbalizer = self.delete_tokens(sign.ques + self.verbalize_field("integer", self.graph))

    def build_graphs(self):
        ds = delete(self.SPACE).star
        digit = spoken_map(load_pairs("numbers/digit.tsv"))
        tens = spoken_map(load_pairs("numbers/tens.tsv"))
        zero = spoken_map(load_pairs("numbers/zero.tsv"))
        self.digit = digit
        self.zero = zero

        two_nonzero = (tens | insert("०") + digit).optimize()
        two = (two_nonzero | insert("००")).optimize()
        hundreds = (digit + delete("शे") + ds | insert("०")) + two
        # ``एकतीसशे`` (thirty-one hundred) fills the thousands and hundreds slots.
        hundreds_count = insert("०") + tens + delete("शे") + ds + two
        thousands_and_hundreds = []
        higher = pynini.accep("")
        for scale in SCALES:
            slot = two_nonzero + ds + delete(scale) + ds | add_weight(insert("००"), SKIP_WEIGHT)
            if scale == "हजार":
                thousands_and_hundreds = slot + hundreds | hundreds_count
            else:
                higher += slot
        nonzero = DIGIT.star + NON_ZERO + DIGIT.star
        strip_zeros = delete("०").star + NON_ZERO + DIGIT.star

        trimmed = self.NOT_SPACE | self.NOT_SPACE + self.VSIGMA + self.NOT_SPACE
        trimmed = trimmed.optimize()
        hundred = pynini.cross("शंभर", "१००")
        graph = (higher + thousands_and_hundreds) @ strip_zeros | hundred | zero
        self.graph = (trimmed @ graph).optimize()
        # 1-999 (including ``शंभर``) counts a lakh/crore quantity: ``सतरा लाख``.
        self.hundreds = (trimmed @ ((hundreds @ nonzero) @ strip_zeros | hundred)).optimize()
