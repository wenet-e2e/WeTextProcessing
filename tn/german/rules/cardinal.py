# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict

import pynini
from pynini.lib import pynutil

from tn.processor import Processor
from tn.utils import get_abs_path, load_labels


TOKEN_ORDER = {}


def _compound_tens():
    """Build the German units-before-tens transducer from numeric resources."""
    digits = defaultdict(list)
    ties = defaultdict(list)
    for word, number in load_labels(get_abs_path("german/data/numbers/digit.tsv")):
        digits[number].append(word)
    digits["1"] = ["ein"]
    for word, number in load_labels(get_abs_path("german/data/numbers/ties.tsv")):
        ties[number].append(word)
    return pynini.string_map([
        (str(number), digit + " und " + tie)
        for number in range(21, 100) if number % 10
        for digit in digits[str(number)[1]]
        for tie in ties[str(number)[0]]
    ])


class Cardinal(Processor):
    """Raw German cardinal fields, with all numeric conversion in verbalization."""

    def __init__(self, deterministic=True):
        super().__init__("cardinal", token_orders=TOKEN_ORDER)
        self.deterministic = deterministic

        def number_table(name):
            return pynini.string_file(get_abs_path("german/data/numbers/" + name + ".tsv")).invert()

        zero = number_table("zero")
        digit_no_one = number_table("digit")
        digit = digit_no_one | number_table("ones")
        self.digits = (digit | zero).optimize()
        teen = number_table("teen")
        ties = number_table("ties")
        tens = (pynutil.delete("0") + digit | _compound_tens() | teen | ties + pynutil.delete("0"))
        hundred = (
            (digit_no_one + pynutil.insert(" ") | pynini.cross("1", "ein "))
            + pynutil.insert("hundert")
            + (
                pynini.closure(pynutil.insert(" ") + pynutil.insert("und", weight=0.0001), 0, 1)
                + pynutil.insert(" ") + tens
                | pynutil.delete("00")
            )
            | pynutil.delete("0") + tens
        ).optimize()
        thousand = (
            (hundred + pynutil.insert(" tausend") | pynutil.delete("000"))
            + (pynutil.insert(" ") + hundred | pynutil.delete("000"))
        )

        def quantity(word, plural):
            return (
                hundred + pynutil.insert(" " + word)
                + pynini.closure(pynutil.insert(plural, weight=-0.0001), 0, 1)
                | pynutil.delete("000")
            )

        groups = [
            quantity("trilliarde", "n"),
            quantity("trillion", "en"),
            quantity("billiarde", "n"),
            quantity("billion", "en"),
            quantity("milliarde", "n"),
            quantity("million", "en"),
        ]
        grouped = groups[0]
        for group in groups[1:]:
            grouped = grouped + pynutil.insert(" ") + group
        grouped = grouped + pynutil.insert(" ") + thousand

        positive_not_one = ((self.DIGIT - "0") + pynini.closure(self.DIGIT)) - "1"
        pad = pynini.cdrewrite(pynini.closure(pynutil.insert("0")), "[BOS]", "", self.VSIGMA)
        fix_syntax = pynini.string_map([
            ("eins tausend", "ein tausend"),
            ("eins millionen", "eine million"),
            ("eins milliarden", "eine milliarde"),
            ("eins billionen", "eine billion"),
            ("eins billiarden", "eine billiarde"),
        ])
        self.graph = (
            positive_not_one @ pad @ pynini.closure(self.DIGIT, 24, 24) @ grouped
            @ pynini.cdrewrite(self.DELETE_SPACE, "[BOS]", "", self.VSIGMA)
            @ pynini.cdrewrite(self.DELETE_SPACE, "", "[EOS]", self.VSIGMA)
            @ pynini.cdrewrite(pynini.cross("  ", " "), "", "", self.VSIGMA)
            @ pynini.cdrewrite(fix_syntax, "[BOS]", "", self.VSIGMA)
            | zero | pynini.cross("1", "eins")
        ).optimize()
        self.hundreds = (
            positive_not_one @ pad @ pynini.closure(self.DIGIT, 3, 3) @ hundred
            | pynini.cross("1", "eins")
        ).optimize()
        self.two_digits = (pynini.closure(self.DIGIT, 1, 2) @ self.hundreds).optimize()
        self.sign = pynini.cross("-", "minus ")

        sign_tagger = pynini.closure(self.tag_field("negative", self.sign) + pynutil.insert(" "), 0, 1)
        sign_verbalizer = pynini.closure(self.verbalize_field("negative", self.sign) + pynutil.delete(" "), 0, 1)
        self.tagger = self.add_tokens(sign_tagger + self.tag_field("integer", self.graph))
        self.verbalizer = self.delete_tokens(sign_verbalizer + self.verbalize_field("integer", self.graph))
