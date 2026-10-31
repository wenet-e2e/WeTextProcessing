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
from pynini.lib import utf8
from pynini.lib.pynutil import delete, insert

from tn.marathi.rules.utils import MR_DIGIT, MR_NON_ZERO, canonical, number_words, to_devanagari, written_map
from tn.processor import Processor

# Indian place values above the hundreds, each spanning two digits.
SCALES = ("अब्ज", "कोटी", "लाख", "हजार")
WIDTH = 3 + 2 * len(SCALES)
MINUS = "उणे"


class Cardinal(Processor):
    """Written cardinals, e.g. ``-७२२`` or ``-722`` -> ``उणे सातशे बावीस``.

    The tagger keeps the written digits (and the ``-`` sign); the verbalizer
    reads them with Indian grouping up to ``अब्ज`` (10^9). Comma-grouped
    numbers (``१,००,०००``) are read as one number, and numbers with leading
    zeros or more than eleven digits are read digit by digit.
    """

    def __init__(self):
        super().__init__("cardinal", token_orders={})
        self.build_graphs()
        negative = self.tag_field("negative", pynini.accep("-")) + insert(" ")
        self.tagger = self.add_tokens(negative.ques + self.tag_field("integer", self.graph))
        sign = self.verbalize_field("negative", pynini.cross("-", MINUS + " ")) + delete(" ")
        self.verbalizer = self.delete_tokens(sign.ques + self.verbalize_field("integer", self.graph))

    def build_graphs(self):
        words = number_words()
        self.digit = written_map([(to_devanagari(n), words[n]) for n in range(1, 10)])
        self.zero = written_map([(to_devanagari(0), words[0])])
        # Two-digit groups with at least one non-zero digit, e.g. ``०७`` or ``४२``.
        self.two = written_map([(to_devanagari(n, 2), words[n]) for n in range(1, 100)])
        self.tens = written_map([(to_devanagari(n), words[n]) for n in range(10, 100)])
        hundreds = self.digit + insert("शे") + (delete("००") | insert(" ") + self.two) | delete("०") + self.two
        self.hundreds = hundreds.optimize()

        fixed = pynini.accep("")
        for scale in SCALES:
            fixed += insert(" ") + self.two + insert(" " + scale) | delete("००")
        fixed += insert(" ") + self.hundreds | delete("०००")
        pad = pynini.union(*[insert("०" * (WIDTH - k)) + MR_NON_ZERO + MR_DIGIT ** (k - 1) for k in range(1, WIDTH + 1)])
        lstrip = delete(" ") + utf8.VALID_UTF8_CHAR.star
        number = pad @ fixed @ lstrip
        # ``शंभर`` is a hundred on its own; inside larger numbers it is ``एकशे``.
        number = pynini.difference(MR_DIGIT.plus, "१००") @ number | pynini.cross("१००", "शंभर")
        self.number = (number | self.zero).optimize()

        single = (self.digit | self.zero).optimize()
        self.digits = (single + (insert(" ") + single).star).optimize()
        long_or_padded = MR_DIGIT ** (WIDTH + 1) + MR_DIGIT.star | "०" + MR_DIGIT.plus
        digitwise = long_or_padded @ self.digits

        comma = delete(",")
        indian = MR_DIGIT.ques + MR_NON_ZERO + (comma + MR_DIGIT ** 2).star + comma + MR_DIGIT ** 3
        western = pynini.closure(MR_DIGIT, 0, 2) + MR_NON_ZERO + (comma + MR_DIGIT ** 3).plus
        grouped = (indian | western) @ self.number
        self.graph = canonical(self.number | grouped | digitwise)
        self.integer = canonical(self.number | grouped)
