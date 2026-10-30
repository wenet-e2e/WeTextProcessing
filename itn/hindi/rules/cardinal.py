# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from itn.hindi.rules.utils import DEDH, DELETE_SPACE, HI_DIGIT, MINUS, spoken_to_written, string_file
from tn.processor import Processor

TOKEN_ORDER = {}

HUNDRED = "सौ"
THOUSAND = pynini.union("हज़ार", "हजार").optimize()


def fractional_hundreds(prefix, graph, suffix, scale):
    """``साढ़े तीन सौ`` style readings: drop the prefix and the scale word, append digits."""

    return add_weight(delete(prefix) + DELETE_SPACE + graph + insert(suffix, weight=-0.1) + DELETE_SPACE
                      + delete(scale), -0.1)


class Cardinal(Processor):
    """``ऋणात्मक तेईस`` -> ``cardinal { negative: "ऋणात्मक" integer: "तेईस" }`` -> ``-२३``."""

    def __init__(self):
        super().__init__("cardinal", ordertype="itn", token_orders=TOKEN_ORDER)
        zero = spoken_to_written("numbers/zero.tsv")
        digit = spoken_to_written("numbers/digit.tsv")
        teens_and_ties = spoken_to_written("numbers/teens_and_ties.tsv")
        paune = spoken_to_written("numbers/paune.tsv")
        self.zero = zero
        self.digit = digit
        self.teens_and_ties = teens_and_ties
        self.paune = paune
        self.single_digit_with_zero = (insert("०") + digit).optimize()
        self.two_digit = (teens_and_ties | insert("०") + digit).optimize()

        hundred_component = pynini.union(digit + DELETE_SPACE + delete(HUNDRED), insert("०"))
        hundred_component += DELETE_SPACE + (self.two_digit | insert("००"))

        # "eleven hundred" style hundreds and the quarter readings of thousands.
        hundred_as_thousand = pynini.union(teens_and_ties + DELETE_SPACE + delete(HUNDRED), insert("०"))
        hundred_as_thousand += DELETE_SPACE + (self.two_digit | insert("००"))
        hundred_as_thousand |= fractional_hundreds("साढ़े", digit, "५००", THOUSAND)
        hundred_as_thousand |= fractional_hundreds("सवा", digit, "२५०", THOUSAND)
        hundred_as_thousand |= fractional_hundreds("पौने", paune, "७५०", THOUSAND)
        hundred_as_thousand |= fractional_hundreds(DEDH, insert(""), "१५००", THOUSAND)
        hundred_as_thousand |= fractional_hundreds("ढाई", insert(""), "२५००", THOUSAND)

        in_hundreds = fractional_hundreds("साढ़े", digit | self.two_digit, "५०", HUNDRED)
        in_hundreds |= fractional_hundreds("सवा", digit | self.two_digit, "२५", HUNDRED)
        in_hundreds |= fractional_hundreds("पौने", paune, "७५", HUNDRED)
        in_hundreds |= fractional_hundreds(DEDH, insert(""), "१५०", HUNDRED)
        in_hundreds |= fractional_hundreds("ढाई", insert(""), "२५०", HUNDRED)
        hundreds = hundred_component | hundred_as_thousand | in_hundreds

        # Indian numbering system: shankh, padma, nil, kharab, arab, crore, lakh, thousand.
        def scale(word):
            return pynini.union(self.two_digit + DELETE_SPACE + delete(word), insert("००", weight=0.1))

        crores = pynini.union(self.two_digit + DELETE_SPACE + delete("करोड़") | delete("crores"),
                              insert("००", weight=0.1))
        thousands = pynini.union(self.two_digit + DELETE_SPACE + delete(THOUSAND), insert("००", weight=0.1))
        indian = (scale("शंख") + DELETE_SPACE + scale("पद्म") + DELETE_SPACE + scale("नील") + DELETE_SPACE
                  + scale("खरब") + DELETE_SPACE + scale("अरब") + DELETE_SPACE + crores + DELETE_SPACE
                  + scale("लाख") + DELETE_SPACE + thousands)
        no_prefix = add_weight(pynini.cross(HUNDRED, "१००") | pynini.cross(THOUSAND, "१०००")
                               | pynini.cross("लाख", "१०००००") | pynini.cross("करोड़", "१०००००००"), 2)

        graph = pynini.union(indian + DELETE_SPACE + hundreds, zero, no_prefix)
        graph @= pynini.union(delete("०").star + (HI_DIGIT - "०") + HI_DIGIT.star, "०")
        self.graph_no_exception = graph.optimize()
        exceptions = string_file("numbers/labels_exception.tsv")
        self.graph = ((self.graph_no_exception.copy().project("input") - exceptions.arcsort())
                      @ self.graph_no_exception).optimize()

        self.tagger = self.add_tokens((self.tag_field("negative", MINUS) + delete(" ") + insert(" ")).ques
                                      + self.tag_field("integer", self.graph))
        negative = self.verbalize_field("negative", pynini.cross(MINUS, "-")) + delete(" ")
        self.verbalizer = self.delete_tokens(negative.ques + self.verbalize_field("integer", self.graph))
