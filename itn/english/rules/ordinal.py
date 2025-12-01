# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
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

from pynini import closure, cross, string_file, union
from pynini.lib.pynutil import insert

from itn.english.rules.cardinal import Cardinal
from tn.processor import Processor
from tn.utils import get_abs_path


class Ordinal(Processor):
    """
    Finite state transducer for classifying ordinal
        e.g. thirteenth -> ordinal { integer: "13" }

    Args:
        cardinal: CardinalFst
        input_case: accepting either "lower_cased" or "cased" input.
    """

    def __init__(self):
        super().__init__("ordinal")
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        graph_digit = string_file(get_abs_path("../itn/english/data/ordinals/digit.tsv"))
        graph_teens = string_file(get_abs_path("../itn/english/data/ordinals/teen.tsv"))
        graph = closure(self.CHAR) + union(graph_digit, graph_teens, cross("tieth", "ty"), cross("th", ""))

        self.graph = graph @ Cardinal().graph_no_exception
        self.graph |= ((self.TO_LOWER + self.SIGMA) @ self.graph).optimize()

        convert_eleven = cross("11", "11th")
        convert_twelve = cross("12", "12th")
        convert_thirteen = cross("13", "13th")
        convert_one = cross("1", "1st")
        convert_two = cross("2", "2nd")
        convert_three = cross("3", "3rd")
        convert_rest = insert("th", weight=0.01)
        suffix =  self.build_rule(
            convert_eleven
            | convert_twelve
            | convert_thirteen
            | convert_one
            | convert_two
            | convert_three
            | convert_rest,
            "",
            "[EOS]",
        )
        self.graph = self.graph @ suffix

        tagger = insert('value: "') + self.graph + insert('"')
        self.tagger = self.add_tokens(tagger).optimize()
