# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

from itn.hebrew.rules.utils import (ALPHA_HE, DELETE_SPACE, DIGIT, PREFIX, SPACE, VSIGMA, first_field, load_labels,
                                    string_file, trim)
from tn.processor import Processor

DELETE_AND = delete("ו")
DELETE_OPTIONAL_AND = DELETE_AND.ques
NONZERO = (DIGIT.star + (DIGIT - "0") + DIGIT.star).optimize()
STRIP_ZEROS = (delete("0").star + (DIGIT - "0") + DIGIT.star | "0").optimize()
GROUP_BY_THREES = (pynini.closure(DIGIT, 1, 3) + (insert(",") + DIGIT**3).star).optimize()


class Cardinal(Processor):
    """Spoken cardinals, e.g. ``כמאה עשרים וחמש`` -> ``כ-125``.

    The tagger keeps the spoken words (and an attached one-letter prefix in
    its own field); the verbalizer assembles the digits in comma-separated
    groups of three.
    """

    def __init__(self):
        super().__init__(name="cardinal", ordertype="itn", token_orders={})
        self.build_graphs()
        negative = self.tag_field("negative", pynini.accep("מינוס")) + delete(SPACE.plus) + insert(" ")
        number = (negative + self.tag_field("integer", self.graph)
                  | negative.ques + self.tag_field("integer", self.graph_wo_small))
        self.tagger = self.add_tokens((self.tag_field("prefix", PREFIX) + insert(" ")).ques + number)

        self.verbalizer = self.delete_tokens(
            first_field(self, "prefix", PREFIX + insert("-")).ques
            + first_field(self, "negative", pynini.cross("מינוס", "-")).ques
            + self.verbalize_field("integer", self.graph @ GROUP_BY_THREES))

    def build_graphs(self):
        zero = string_file("numbers/zero.tsv")
        digit = string_file("numbers/digit.tsv")
        teen = string_file("numbers/teen.tsv")
        ties = string_file("numbers/ties.tsv")
        ties += DELETE_SPACE + DELETE_OPTIONAL_AND + digit | insert("0", weight=0.001)
        two_digit = teen | ties
        self.graph_two_digit = (digit | ties | add_weight(teen, -0.001)).optimize()

        hundred = DELETE_OPTIONAL_AND + (
            string_file("numbers/hundreds_exception.tsv")
            | digit + DELETE_SPACE + delete("מאות")
            | insert("0", weight=0.001))
        hundred += DELETE_SPACE
        hundred += (DELETE_OPTIONAL_AND + two_digit
                    | insert("0") + DELETE_SPACE + DELETE_AND + digit
                    | insert("00", weight=0.001))
        hundred = (hundred | insert("0") + two_digit | insert("00") + digit).optimize()
        self.graph_hundred = trim(hundred @ NONZERO @ STRIP_ZEROS)

        large_prefix = hundred | insert("0") + two_digit | insert("00") + string_file("numbers/thousands.tsv")
        delete_thousand = delete("אלפים") | delete("אלף", weight=0.001)
        thousands = DELETE_OPTIONAL_AND + (
            insert("00") + string_file("numbers/thousands_exception.tsv")
            | large_prefix + DELETE_SPACE + delete_thousand
            | insert("000", weight=0.001))
        graph_thousands = thousands + DELETE_SPACE + hundred | zero
        self.graph_thousands = trim(graph_thousands @ STRIP_ZEROS)

        delete_million = delete("מיליונים") | delete("מיליון", weight=0.001)
        millions = (large_prefix + DELETE_SPACE + delete_million
                    | insert("00") + string_file("numbers/millions_exception.tsv")
                    | insert("000", weight=0.001))
        graph = millions + DELETE_SPACE + thousands + DELETE_SPACE + hundred | zero
        # A leading ``ו`` is the conjunction prefix, never part of the number.
        graph = ((ALPHA_HE - "ו") + VSIGMA) @ graph @ STRIP_ZEROS
        self.graph = trim(graph)

        # Bare digits, ten and zero stay words unless negated or quantified.
        small = sorted({row[0] for row in load_labels("numbers/digit.tsv")} | {"אפס", "עשר", "עשרה"})
        small = pynini.union(*small, *["ו" + word for word in small]).optimize()
        self.graph_wo_small = ((pynini.project(self.graph, "input") - small) @ self.graph).optimize()
        hours = pynini.union(*{row[0] for row in load_labels("numbers/viable_hours.tsv")}).optimize()
        self.graph_wo_viable_hours = ((pynini.project(self.graph, "input") - hours) @ self.graph).optimize()
