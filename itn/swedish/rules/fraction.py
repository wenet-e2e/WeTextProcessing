# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pynini.lib import pynutil

from itn.swedish.rules.ordinal import number_graph, ordinal_words
from tn.processor import Processor

TOKEN_ORDER = {"fraction": ["negative", "integer_part", "numerator", "denominator"]}

_ENDINGS = (
    ("ljarte", "ljarddel"), ("tionde", "tiondel"), ("tonde", "tondel"), ("ljonte", "ljondel"),
    ("lliarte", "lliarddel"), ("llionte", "lliondel"), ("tusende", "tusendel"), ("te", "tedel"),
    ("de", "dedel"), ("je", "jedel"), ("drade", "dradel"), ("a", "adel"),
)


def fraction_word(ordinal):
    """Applies the rightmost-anchored, leftmost-starting ordinal-to-fraction ending."""

    matches = [(len(old), old, new) for old, new in _ENDINGS if ordinal.endswith(old)]
    if not matches:
        return ordinal
    _, old, new = max(matches)
    return ordinal[:-len(old)] + new


def forms(word):
    return [word, word[:-3] + "delar"] if word.endswith("del") else [word]


class Fraction(Processor):
    """Raw spoken fractions (``tjugotre och fyra femtedelar``) to ``23 4/5``."""

    def __init__(self, cardinal):
        super().__init__(name="fraction", ordertype="itn", token_orders=TOKEN_ORDER)

        def convert(pairs):
            return [(form, value) for word, value in pairs for form in forms(fraction_word(word))]

        digits, teens, ties = ordinal_words()

        def scale_word(scale):
            return pynini.union(*forms(fraction_word(scale + "te" if scale.endswith("n") else scale[:-1] + "te")))

        hundredth = pynini.union(*forms(fraction_word("hundrade")))
        thousandth = pynini.union(*forms(fraction_word("tusende")))
        denominators = number_graph(
            cardinal, convert(digits), convert(teens), convert(ties), hundredth, thousandth, scale_word)
        lexicalised = pynini.union("andradel", "andradelar", "fjärdedel", "fjärdedelar")
        denominators = pynini.difference(self.input_projection(denominators), lexicalised) @ denominators
        denominators |= pynini.string_map([("halv", "2"), ("halva", "2"), ("kvart", "4"), ("kvartar", "4")])
        self.denominators = denominators.optimize()

        number = cardinal.graph
        negative = cardinal.negative
        sign_t = (self.tag_field("negative", negative) + pynini.accep(" ")).ques
        sign_v = (self.verbalize_field("negative", negative) + pynutil.delete(" ")).ques
        integer_t = self.tag_field("integer_part", number)
        numerator_t = self.tag_field("numerator", number)
        denominator_t = self.tag_field("denominator", self.denominators)
        och = pynutil.delete("och") + pynini.accep(" ")
        integer_t = sign_t + integer_t + pynini.accep(" ")
        tagger = (
            integer_t + och + denominator_t
            | (integer_t + och.ques).ques + numerator_t + pynini.accep(" ") + denominator_t
        )
        self.tagger = self.add_tokens(tagger)

        integer_v = self.verbalize_field("integer_part", number) + pynutil.delete(" ") + pynutil.insert(" ")
        numerator_v = self.verbalize_field("numerator", number) + pynutil.delete(" ")
        verbalizer = (
            (sign_v + integer_v).ques
            + (numerator_v | pynutil.insert("1"))
            + pynutil.insert("/") + self.verbalize_field("denominator", self.denominators)
        )
        self.verbalizer = self.delete_tokens(verbalizer)
