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

from itn.spanish.rules.utils import DELETE_SPACE, DIGIT, MINUS, cased, data_file, restrict
from tn.processor import Processor

TOKEN_ORDER = {"cardinal": ["negative", "integer"]}


class Cardinal(Processor):
    """Raw spoken cardinals below 10^24; V writes the digits.

    Numbers below ten stay words ("vivo en una casa").
    """

    def __init__(self, cased_input=False):
        super().__init__("cardinal", ordertype="itn", token_orders=TOKEN_ORDER)
        self.cased_input = cased_input
        word = pynutil.delete
        zero = data_file("numbers/zero.tsv")
        digit = data_file("numbers/digit.tsv")
        ties = data_file("numbers/ties.tsv")
        teen = data_file("numbers/teen.tsv")
        twenties = data_file("numbers/twenties.tsv")
        hundreds = data_file("numbers/hundreds.tsv")

        # As upstream, "y" is accepted before any digit ("mil y uno").
        full_ties = (ties | pynutil.insert("0")) + (
            DELETE_SPACE + word("y") + DELETE_SPACE + digit | pynutil.insert("0")
        )
        component = (hundreds | pynutil.insert("0")) + DELETE_SPACE + pynini.union(
            twenties, full_ties, teen, pynutil.insert("0") + digit, pynutil.insert("00"),
        )
        nonzero = (component @ (pynini.closure(DIGIT) + (DIGIT - "0") + pynini.closure(DIGIT))).optimize()
        self.hundreds = nonzero
        mil = nonzero + DELETE_SPACE + word("mil") | pynutil.insert("001") + word("mil")
        thousands = mil | pynutil.insert("000", weight=0.1)

        def power(singular, plural):
            return nonzero + DELETE_SPACE + (word(plural) | word(singular))

        millions = power("millón", "millones") | pynutil.insert("000") + word("millones")
        mil_millions = mil + DELETE_SPACE + (millions | pynutil.insert("000") + word("millones"))
        mil_millions |= pynutil.insert("000000", weight=0.1)
        millardo = nonzero + DELETE_SPACE + (word("millardo") | word("millardos"))
        billions = power("billón", "billones")
        mil_billions = mil + DELETE_SPACE + (billions | pynutil.insert("000") + word("billones"))
        mil_billions |= pynutil.insert("000000", weight=0.1)
        trillions = power("trillón", "trillones")
        mil_trillions = mil + DELETE_SPACE + (trillions | pynutil.insert("000") + word("trillones"))
        mil_trillions |= pynutil.insert("000000", weight=0.1)

        graph = (
            (mil_trillions | pynutil.insert("000", weight=0.1) + trillions) + DELETE_SPACE
            + (mil_billions | pynutil.insert("000", weight=0.1) + billions) + DELETE_SPACE
            + pynini.union(
                mil_millions,
                pynutil.insert("000", weight=0.1) + millions,
                millardo + millions,
                millardo + pynutil.insert("000", weight=0.1),
            )
            + DELETE_SPACE + thousands + DELETE_SPACE + component
        ) | zero
        graph @= pynutil.delete(pynini.closure("0")) + (DIGIT - "0") + pynini.closure(DIGIT) | "0"
        graph = graph.optimize()
        # Lower-case words to digits, for composing other word graphs onto.
        self.words = graph
        # Words to digits including zero to nine (for decimals and fractions).
        self.graph_no_exception = cased(graph, cased_input)
        self.numbers_up_to_thousand = cased((graph @ pynini.closure(DIGIT, 1, 3)).optimize(), cased_input)
        self.numbers_up_to_million = cased((graph @ pynini.closure(DIGIT, 1, 6)).optimize(), cased_input)
        self.digit = cased(digit | zero, cased_input)
        # Zero to nine stay words.
        exceptions = pynini.closure(" ", 0, 1) + self.digit
        self.graph = restrict(self.graph_no_exception, exceptions)

        self.sign = cased(pynini.cross(MINUS, "-"), cased_input)
        sign_t = pynini.closure(self.tag_field("negative", self.sign) + pynini.accep(" "), 0, 1)
        sign_v = pynini.closure(self.verbalize_field("negative", self.sign) + pynutil.delete(" "), 0, 1)
        self.sign_tagger = sign_t
        self.sign_verbalizer = sign_v
        self.tagger = self.add_tokens(sign_t + self.tag_field("integer", self.graph))
        self.verbalizer = self.delete_tokens(sign_v + self.verbalize_field("integer", self.graph))
