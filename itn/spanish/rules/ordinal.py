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

from itn.spanish.rules.utils import DELETE_SPACE, SIGMA, cased, data_file, restrict
from tn.processor import Processor

TOKEN_ORDER = {}


class Ordinal(Processor):
    """Raw spoken ordinals below 1000; V writes digits and the gender mark.

    First to ninth stay words ("primero hice", "segunda guerra mundial").
    """

    def __init__(self, cardinal, cased_input=False):
        super().__init__("ordinal", ordertype="itn", token_orders=TOKEN_ORDER)
        digit = data_file("ordinals/digit.tsv")
        union = pynini.union(
            digit,
            data_file("ordinals/teen.tsv"),
            data_file("ordinals/twenties.tsv"),
            data_file("ordinals/ties.tsv")
            | data_file("ordinals/ties.tsv") + pynini.cross(" ", "y") + digit,
            data_file("ordinals/hundreds.tsv"),
        ).optimize()
        # Leading ordinal words are always part of the number.
        front = pynini.closure(pynutil.add_weight(union, -0.1) + DELETE_SPACE)
        graphs = {}
        for ending, mark in (("o", ".º"), ("a", ".ª"), ("er", ".ᵉʳ")):
            graph = (front + ((SIGMA + ending) @ union)) @ cardinal.words
            graphs[ending] = (graph + pynutil.insert(mark)).optimize()
        # Masculine ordinal words to digits, used for fraction denominators.
        self.masc = (front + ((SIGMA + "o") @ union)) @ cardinal.words
        self.masc = self.masc.optimize()
        self.graph = restrict(cased(pynini.union(*graphs.values()), cased_input), cased(digit, cased_input))
        self.tagger = self.add_tokens(self.tag_field("integer", self.graph))
        self.verbalizer = self.delete_tokens(self.verbalize_field("integer", self.graph))
