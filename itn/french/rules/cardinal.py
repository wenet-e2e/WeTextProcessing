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
from pynini.lib import utf8
from pynini.lib.pynutil import delete, insert

from itn.utils import get_abs_path
from tn.processor import Processor


TOKEN_ORDER = {"cardinal": ["negative", "integer"]}


def load(name):
    return pynini.string_file(get_abs_path("french/data/{}.tsv".format(name)))


class Cardinal(Processor):
    """Spoken cardinals stay verbatim in the tag; digits appear on verbalization.

    ``cent cinquante et un`` -> ``cardinal { integer: "cent cinquante et un" }`` -> ``151``.
    Both the reformed orthography (hyphens everywhere) and the traditional one
    (spaces around "cent", "mille", "million", "et", ...) are accepted, up to
    the trilliards. Numbers below ten are left as words.
    """

    def __init__(self):
        super().__init__("cardinal", ordertype="itn", token_orders=TOKEN_ORDER)
        hyphen = delete("-").ques
        self.hyphen = hyphen
        zero = load("numbers/zero")
        digit = load("numbers/digit")
        teens = load("numbers/teen")
        ties = load("numbers/ties")
        ties_unique = load("numbers/ties_unique")

        tens = ties + (hyphen + digit | insert("0")) | teens | ties_unique
        tens_with_zeros = tens | insert("0") + (digit | insert("0", weight=0.01))

        cent = delete("cent")
        cents = pynini.cross("cents", "00")
        digit_no_one = (self.input_projection(digit) - pynini.union("un", "une")) @ digit
        hundreds_singular = (digit_no_one + hyphen + cent | pynini.cross("cent", "1")) + hyphen + tens_with_zeros
        hundreds_plural = digit_no_one + hyphen + cents
        hundreds = hundreds_singular | hundreds_plural | insert("0") + tens_with_zeros
        nonzero = self.DIGIT.star + (self.DIGIT - "0") + self.DIGIT.star
        hundreds_nonzero = (hundreds @ nonzero).optimize()
        self.hundreds_nonzero = self.rewrite(hundreds_nonzero)

        tens_of_hundreds = tens + hyphen + cent + hyphen + tens_with_zeros | tens + hyphen + cents

        def magnitude(*names, skip=True):
            graph = hundreds_nonzero + hyphen + pynini.union(*[delete(name) for name in names])
            return graph | insert("000", weight=0.1) if skip else graph

        thousands = magnitude("mille", skip=False) | insert("001") + delete("mille") | insert("000", weight=0.1)
        millions = magnitude("million", "millions")
        milliards = magnitude("milliard", "milliards")
        billions = magnitude("billions", "billion")
        billiards = magnitude("billiards", "billiard")
        trillions = magnitude("trillions", "trillion")
        trilliards = magnitude("trilliards", "trilliard")

        graph = trilliards
        for component in (trillions, billiards, billions, milliards, millions, thousands, hundreds):
            graph = graph + hyphen + component
        graph = (graph | tens_of_hundreds | zero).optimize()
        strip = delete(pynini.accep("0").star) + (self.DIGIT - "0") + self.DIGIT.star | "0"
        graph = self.rewrite((graph @ strip).optimize())

        self.graph_no_exception = graph
        self.zero = zero
        self.numbers_up_to_thousand = (graph @ pynini.closure(self.DIGIT, 1, 3)).optimize()
        self.numbers_up_to_million = (graph @ pynini.closure(self.DIGIT, 1, 6)).optimize()
        exceptions = self.input_projection(digit | zero)
        self.graph = ((self.input_projection(graph) - exceptions) @ graph).optimize()
        self.sign = pynini.cross("moins", "-")

        sign = self.tag_field("negative", self.sign) + delete(" ") + insert(" ")
        self.tagger = self.add_tokens(sign.ques + self.tag_field("integer", self.graph))
        sign = self.verbalize_field("negative", self.sign) + delete(" ")
        self.verbalizer = self.delete_tokens(sign.ques + self.verbalize_field("integer", self.graph))

    @staticmethod
    def rewrite(graph):
        """Adds the traditional orthography to a hyphenated-words -> digits graph.

        A traditional spelling has a space wherever the reformed spelling has a
        hyphen next to "et", "cent", "mille", "million", ... and no hyphen next
        to these words (mixed spellings such as "deux-cent neuf" are rejected).
        Each rewrite is composed onto the finite inverted number graph, so the
        context-dependent rules never interact over an unbounded sigma-star.
        """

        targets = pynini.union(
            "et", "cent", "mille", "million", "milliard", "billion", "billiard", "trillion", "trilliard"
        ) + pynini.accep("s").ques
        sigma = utf8.VALID_UTF8_CHAR.star
        before = pynini.cdrewrite(pynini.cross("-", " "), "", targets, sigma)
        after = pynini.cdrewrite(pynini.cross("-", " "), targets, "", sigma)
        spoken = ((pynini.invert(graph) @ before).optimize() @ after).optimize()
        return (graph | pynini.invert(spoken)).optimize()
