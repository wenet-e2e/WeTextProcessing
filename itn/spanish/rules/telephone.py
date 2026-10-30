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

from itn.spanish.rules.utils import cased, data_file, data_labels
from tn.processor import Processor

TOKEN_ORDER = {}


class Telephone(Processor):
    """Raw spoken phone numbers of 8, 9 or 10 digits; V groups the digits.

    Digits are read one by one or in pairs ("doce treinta y cuatro"), with
    "doble"/"triple" repetitions, an optional "más" country code and an
    optional "extensión".
    """

    def __init__(self, cased_input=False):
        super().__init__("telephone", ordertype="itn", token_orders=TOKEN_ORDER)
        digit = data_file("numbers/digit.tsv")
        zero = pynini.cross("cero", "0")
        single = (digit | zero).optimize()
        double = pynini.union(
            data_file("numbers/twenties.tsv"),
            data_file("numbers/teen.tsv"),
            data_file("numbers/ties.tsv") + pynutil.insert("0"),
            data_file("numbers/ties.tsv") + pynutil.delete(" y ") + digit,
        )
        delete_space = pynutil.delete(" ")
        twice = single + delete_space + single
        thrice = twice + delete_space + single
        words = data_labels("numbers/digit.tsv") + [["cero", "0"]]
        doubled = pynini.union(*[pynini.cross("doble " + w, n * 2) for w, n in words])
        tripled = pynini.union(*[pynini.cross("triple " + w, n * 3) for w, n in words])
        two = pynini.union(doubled, twice, double)
        three = pynini.union(tripled, single + delete_space + two)
        four = pynini.union(
            two + delete_space + two,
            tripled + delete_space + single,
            single + delete_space + tripled,
        )
        dash = pynini.cross(" ", "-")
        ten = three + dash + three + dash + four
        nine = three + dash + three + dash + three
        eight = four + dash + four
        country = pynini.cross("más ", "+") + (single | two | three) + dash
        extension = pynini.cross(" extensión ", " ext. ") + (single | two | three)
        graph = (
            pynini.closure(country, 0, 1)
            + pynini.union(pynutil.add_weight(ten, -0.01), nine, eight)
            + pynini.closure(extension, 0, 1)
        )
        self.graph = cased(graph.optimize(), cased_input)
        self.tagger = self.add_tokens(self.tag_field("number_part", self.graph))
        self.verbalizer = self.delete_tokens(self.verbalize_field("number_part", self.graph))
