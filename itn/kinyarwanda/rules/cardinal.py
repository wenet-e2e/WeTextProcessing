# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pynini import accep, cross, difference, invert, union
from pynini.lib.pynutil import add_weight, delete, insert

from itn.kinyarwanda.rules.utils import string_map
from tn.processor import Processor

VOWELS = "aeiou"
CONSONANTS = "bcdfghjklmnpqrstvwxyz"
# Ten has a standalone form and a form that is followed by its units.
TEN = "icumi"
TEN_BEFORE_UNITS = "cumi"
# Scale words: the bare word for exactly one, the class prefix otherwise, and
# the hand-written explicit form for one (``miliyoni imwe``), if any.
SCALES = (
    ("tiriyoni", "tiriyoni ", "tiriyoni imwe"),
    ("miliyoni", "miliyoni ", "miliyoni imwe"),
    ("igihumbi", "ibihumbi ", None),
)
# The explicit forms for one are accepted but never preferred over the bare
# scale word when several spellings are requested.
EXPLICIT_ONE_WEIGHT = 0.01
# TN readings are not unique: ``ibihumbi ijana na mirongo itatu`` is written
# for both 130000 and 100030. Each part costs a little more the lower its
# scale group, so the reading that keeps parts under the larger scale wins.
GROUP_PART_WEIGHT = 0.001


class Cardinal(Processor):
    """Spoken cardinals, e.g. ``ijana na makumyabiri na gatatu`` -> ``123``.

    Accepts the readings written by Kinyarwanda TN: units agree with what
    they count (``kabiri`` alone, ``ibihumbi bibiri``, ``miliyoni ebyiri``)
    and parts are joined with ``na`` before a consonant and ``n'`` before a
    vowel; ``tiriyoni`` names 10^9. Besides the bare ``miliyoni`` and
    ``tiriyoni`` for exactly one, the explicit ``miliyoni imwe`` and
    ``tiriyoni imwe`` are accepted. Digit-by-digit readings that start with
    ``zeru`` (``zeru zeru kabiri``) are written as a digit string.

    The tagger keeps the spoken words (``integer`` or ``digits``); the
    verbalizer writes the digits.
    """

    def __init__(self):
        super().__init__(name="cardinal", ordertype="itn", token_orders={})
        digit = string_map("kinyarwanda/data/numbers/digit.tsv")
        zero = string_map("kinyarwanda/data/numbers/zero.tsv")
        integer = invert(zero | self.build_number()).optimize()
        any_digit = invert(digit | zero)
        digits = invert(zero) + (delete(" ") + any_digit).plus

        tagger = self.tag_field("integer", integer) | self.tag_field("digits", digits.optimize())
        self.tagger = self.add_tokens(tagger)
        verbalizer = self.verbalize_field("integer", integer) | self.verbalize_field("digits", digits.optimize())
        self.verbalizer = self.delete_tokens(verbalizer)

    def connect(self, graph):
        """Prefixes a part with ``na`` or ``n'`` depending on its first letter."""

        vowel_initial = union(*VOWELS) + self.VSIGMA
        consonant_initial = union(*CONSONANTS) + self.VSIGMA
        return (insert(" n'") + (graph @ vowel_initial)) | (insert(" na ") + (graph @ consonant_initial))

    def group(self, units, weight):
        """Maps a three-digit group other than ``000`` to its reading in a unit class.

        Each spoken part of the group costs ``weight``.
        """

        hundreds = string_map("kinyarwanda/data/numbers/hundreds.tsv")
        tens = string_map("kinyarwanda/data/numbers/tens.tsv")
        options = []
        for has_hundreds, has_tens, has_units in product((False, True), repeat=3):
            if not (has_hundreds or has_tens or has_units):
                continue
            ten = cross("1", TEN_BEFORE_UNITS if has_units else TEN) | tens
            graph = accep("")
            started = False
            for present, part in ((has_hundreds, hundreds), (has_tens, ten), (has_units, units)):
                if not present:
                    graph += delete("0")
                    continue
                part = add_weight(part, weight)
                graph += self.connect(part) if started else part
                started = True
            options.append(graph)
        return union(*options).optimize()

    def build_number(self):
        """Builds the digits -> words map; the caller inverts it."""

        digit = string_map("kinyarwanda/data/numbers/digit.tsv")
        digit_bi = string_map("kinyarwanda/data/numbers/digit_bi.tsv")
        digit_e = string_map("kinyarwanda/data/numbers/digit_e.tsv")
        not_one = difference(pynini.closure(self.DIGIT, 3, 3), "001").optimize()

        groups = []
        for index, ((one, prefix, explicit_one), units) in enumerate(zip(SCALES, (digit_e, digit_e, digit_bi))):
            weight = index * GROUP_PART_WEIGHT
            graph = add_weight(cross("001", one), weight) | (not_one @ (insert(prefix) + self.group(units, weight)))
            if explicit_one is not None:
                graph |= add_weight(cross("001", explicit_one), EXPLICIT_ONE_WEIGHT)
            groups.append(graph.optimize())
        groups.append(self.group(digit, len(SCALES) * GROUP_PART_WEIGHT))

        nonzero = difference(self.DIGIT, "0")
        pad = insert("00") + nonzero | insert("0") + nonzero + self.DIGIT | nonzero + self.DIGIT + self.DIGIT
        options = []
        for index, top in enumerate(groups):
            graph = pad @ top
            for lower in groups[index + 1:]:
                graph += delete("000") | self.connect(lower)
            options.append(graph)
        return union(*options).optimize()
