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


TOKEN_ORDER = {"ordinal": ["integer", "noun"]}


class Ordinal(Processor):
    """``vingt-troisièmes`` -> ``ordinal { integer: "vingt-troisièmes" }`` -> ``23ᵉˢ``.

    Before ``siècle`` the number is written in Roman numerals: ``dix-neuvième siècle`` -> ``XIXᵉ siècle``.
    """

    def __init__(self, cardinal):
        super().__init__("ordinal", ordertype="itn", token_orders=TOKEN_ORDER)
        strip = self.VSIGMA + (delete("ième") | load("ordinals/digits_root_change"))
        regular = (strip @ cardinal.graph_no_exception) + insert("e") + pynini.accep("s").ques
        firsts = insert("1") + load("ordinals/firsts")
        second = insert("2") + load("ordinals/second")
        # Spoken ordinal -> digits followed by the plain morphological suffix ("23es").
        ordinal = (regular | firsts | second).optimize()
        nouns = load("ordinals/key_nouns")

        superscript = pynini.union(
            pynini.cross("e", "ᵉ"), pynini.cross("d", "ᵈ"), pynini.cross("r", "ʳ"), pynini.cross("s", "ˢ")).plus
        arabic = (ordinal @ (self.DIGIT.plus + superscript)).optimize()

        roman_digits = load("roman/digits_large").invert()
        roman_ties = load("roman/ties_large").invert()
        roman_hundreds = load("roman/hundreds_large").invert()
        skip = delete("0")
        roman_number = pynini.union(
            self.DIGIT**3 @ (roman_hundreds + (roman_ties | skip) + (roman_digits | skip)),
            self.DIGIT**2 @ (roman_ties + (roman_digits | skip)),
            self.DIGIT @ roman_digits,
        )
        roman = (ordinal @ (roman_number + superscript)).optimize()

        noun = delete(WHITE_SPACE.star) + insert(" ") + self.tag_field("noun", nouns)
        self.tagger = self.add_tokens(self.tag_field("integer", ordinal) | self.tag_field("integer", roman) + noun)

        noun = pynini.accep(" ") + self.verbalize_field("noun", nouns)
        self.verbalizer = self.delete_tokens(
            self.verbalize_field("integer", arabic) | self.verbalize_field("integer", roman) + noun)
