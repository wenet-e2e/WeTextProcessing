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
from pynini.lib.pynutil import add_weight, delete, insert

from itn.utils import get_abs_path
from tn.processor import Processor


TOKEN_ORDER = {}

SCALES = (
    ("milhão", "milhões"),
    ("bilhão", "bilhões"),
    ("trilhão", "trilhões"),
    ("quatrilhão", "quatrilhões"),
    ("quintilhão", "quintilhões"),
    ("sextilhão", "sextilhões"),
)


def number_file(name):
    return pynini.string_file(get_abs_path("portuguese/data/numbers/" + name))


class Cardinal(Processor):
    """Raw spoken cardinals; the verbalizer turns the words into digits.

    The connector "e" is only accepted in its grammatical places (NeMo's
    strict mode), and the words for zero to nine stay unconverted
    ("vivo em uma casa" is not rewritten).
    """

    def __init__(self):
        super().__init__("cardinal", ordertype="itn", token_orders=TOKEN_ORDER)
        zero = number_file("zero.tsv")
        digit = number_file("digit.tsv")
        ties = number_file("ties.tsv")
        teen = number_file("teen.tsv")
        twenties = number_file("twenties.tsv")
        one_hundred = number_file("onehundred.tsv")
        hundreds = number_file("hundreds.tsv")
        self.zero = zero
        self.digit = digit
        self.ties = ties
        self.teen = teen
        self.twenties = twenties

        space = self.DELETE_SPACE
        connector = delete(self.SPACE.plus) + delete("e") + delete(self.SPACE.plus)
        non_zero = self.DIGIT.star + (self.DIGIT - "0") + self.DIGIT.star

        ties_component = pynini.union(
            teen | twenties,
            ties + (connector + digit | insert("0")),
            add_weight(insert("0") + digit, 0.1),
        ) @ non_zero
        hundreds_except_hundred = (pynini.difference(hundreds.copy().project("input"), "cento")) @ hundreds
        prefix_e = (pynini.union(
            one_hundred,
            add_weight(hundreds_except_hundred + insert("00"), 0.1),
            insert("0") + ties_component,
        ) @ non_zero).optimize()
        no_prefix = ((hundreds + connector + ties_component) @ non_zero).optimize()

        # Each magnitude either starts with a group that may follow "e"
        # ("prefix_e": below one hundred or round hundreds) or not.
        lower_prefix_e = (
            (prefix_e + space + delete("mil") | add_weight(insert("001"), 0.1) + delete("mil"))
            + (connector + prefix_e | space + no_prefix | add_weight(insert("000"), 0.1))
        )
        lower_no_prefix = (
            (no_prefix + space + delete("mil") | add_weight(insert("000"), 0.1))
            + (connector + prefix_e | space + no_prefix | add_weight(insert("000"), 0.1))
        )
        tops = [lower_prefix_e]
        for singular, plural in SCALES:
            scale = delete(singular) | delete(plural)
            rest = connector + lower_prefix_e | space + lower_no_prefix
            current_prefix_e = (prefix_e + space + scale + rest).optimize()
            current_no_prefix = ((no_prefix + space + scale | add_weight(insert("000"), 0.1)) + rest).optimize()
            lower_prefix_e, lower_no_prefix = current_prefix_e, current_no_prefix
            tops.append(lower_prefix_e)
        graph = pynini.union(lower_no_prefix, *tops, prefix_e, ties_component, zero)
        strip = delete("0").star + (self.DIGIT - "0") + self.DIGIT.star | "0"
        self.graph_no_exception = (graph @ strip).optimize()

        up_to_thousand = pynini.closure(self.DIGIT, 1, 3)
        up_to_million = pynini.closure(self.DIGIT, 1, 6)
        self.numbers_up_to_thousand = (self.graph_no_exception @ up_to_thousand).optimize()
        self.numbers_up_to_million = (self.graph_no_exception @ up_to_million).optimize()
        years = pynini.union(*[str(year) for year in range(1, 2100)])
        self.digits_from_year = (self.numbers_up_to_million @ years).optimize()

        exception = (digit | zero).project("input")
        self.graph = (
            pynini.difference(self.graph_no_exception.copy().project("input"), exception) @ self.graph_no_exception
        ).optimize()
        self.negative = pynini.cross("menos", "-")

        sign_t = (self.tag_field("negative", self.negative) + delete(self.SPACE.plus) + insert(" ")).ques
        sign_v = (self.verbalize_field("negative", self.negative) + delete(" ")).ques
        self.tagger = self.add_tokens(sign_t + self.tag_field("integer", self.graph))
        self.verbalizer = self.delete_tokens(sign_v + self.verbalize_field("integer", self.graph))
