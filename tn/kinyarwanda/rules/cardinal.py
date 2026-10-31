# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024, DIGITAL UMUGANDA
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

from itertools import product

import pynini
from pynini import accep, cross, difference, union
from pynini.lib.pynutil import add_weight, delete, insert

from tn.kinyarwanda.rules.utils import string_map
from tn.processor import Processor

VOWELS = "aeiou"
CONSONANTS = "bcdfghjklmnpqrstvwxyz"
# Ten has a standalone form and a form that is followed by its units.
TEN = "icumi"
TEN_BEFORE_UNITS = "cumi"
# Scale words: the bare word for exactly one, the class prefix otherwise.
SCALES = (
    ("tiriyoni", "tiriyoni "),
    ("miliyoni", "miliyoni "),
    ("igihumbi", "ibihumbi "),
)


class Cardinal(Processor):
    """Cardinal numbers, e.g. ``123`` -> ``ijana na makumyabiri na gatatu``.

    Every number word agrees with the noun it counts: plain units use the
    counting class (``kabiri``), units of ``ibihumbi`` the bi-class
    (``bibiri``), units of ``miliyoni``/``tiriyoni`` the zi-class
    (``ebyiri``). Parts are joined with ``na`` before a consonant and ``n'``
    before a vowel. As in NeMo, ``tiriyoni`` names 10^9 and numbers up to
    twelve digits are read as a whole; zero-padded, comma-separated or
    longer digit strings are read digit by digit.

    The tagger keeps the written digits (``integer`` for whole numbers,
    ``digits`` for digit-by-digit strings); the verbalizer spells them out.
    """

    def __init__(self):
        super().__init__("cardinal", token_orders={})
        self.number = self.build_number()
        digit = string_map("kinyarwanda/data/numbers/digit.tsv")
        zero = string_map("kinyarwanda/data/numbers/zero.tsv")
        integer = (zero | self.number).optimize()
        any_digit = (digit | zero).optimize()
        digits = (any_digit + (delete(",").ques + insert(" ") + any_digit).star).optimize()

        tagger = self.tag_field("integer", integer) | add_weight(self.tag_field("digits", digits), 0.1)
        self.tagger = self.add_tokens(tagger)
        verbalizer = self.verbalize_field("integer", integer) | self.verbalize_field("digits", digits)
        self.verbalizer = self.delete_tokens(verbalizer)

    def connect(self, graph):
        """Prefixes a part with ``na`` or ``n'`` depending on its first letter."""

        vowel_initial = union(*VOWELS) + self.VSIGMA
        consonant_initial = union(*CONSONANTS) + self.VSIGMA
        return (insert(" n'") + (graph @ vowel_initial)) | (insert(" na ") + (graph @ consonant_initial))

    def group(self, units):
        """Spells a three-digit group other than ``000`` with the given unit class."""

        hundreds = string_map("kinyarwanda/data/numbers/hundreds.tsv")
        tens = string_map("kinyarwanda/data/numbers/tens.tsv")
        options = []
        for has_hundreds, has_tens, has_units in product((False, True), repeat=3):
            if not (has_hundreds or has_tens or has_units):
                continue
            ten = cross("1", TEN_BEFORE_UNITS if has_units else TEN) | tens
            parts = [(has_hundreds, hundreds), (has_tens, ten), (has_units, units)]
            graph = accep("")
            started = False
            for present, part in parts:
                if not present:
                    graph += delete("0")
                    continue
                graph += self.connect(part) if started else part
                started = True
            options.append(graph)
        return union(*options).optimize()

    def build_number(self):
        digit = string_map("kinyarwanda/data/numbers/digit.tsv")
        digit_bi = string_map("kinyarwanda/data/numbers/digit_bi.tsv")
        digit_e = string_map("kinyarwanda/data/numbers/digit_e.tsv")
        three = pynini.closure(self.DIGIT, 3, 3)
        not_one = difference(three, "001").optimize()

        scaled = []
        for (one, prefix), units in zip(SCALES, (digit_e, digit_e, digit_bi)):
            scaled.append((cross("001", one) | (not_one @ (insert(prefix) + self.group(units)))).optimize())
        groups = scaled + [self.group(digit)]

        nonzero = difference(self.DIGIT, "0")
        pad = insert("00") + nonzero | insert("0") + nonzero + self.DIGIT | nonzero + self.DIGIT + self.DIGIT
        options = []
        for index, top in enumerate(groups):
            graph = pad @ top
            for lower in groups[index + 1:]:
                graph += delete("000") | self.connect(lower)
            options.append(graph)
        return union(*options).optimize()
