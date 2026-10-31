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
from pynini.lib.pynutil import add_weight, insert

from itn.utils import get_abs_path
from tn.processor import Processor


TOKEN_ORDER = {}


class Ordinal(Processor):
    """Raw spoken ordinals; the last word's ending selects º or ª.

    Ordinals from one to nine stay unconverted ("segunda guerra mundial").
    """

    def __init__(self):
        super().__init__("ordinal", ordertype="itn", token_orders=TOKEN_ORDER)

        def ordinal_file(name):
            return pynini.string_file(get_abs_path("portuguese/data/ordinals/" + name))

        digit = ordinal_file("digit.tsv")
        ties = ordinal_file("ties.tsv")
        hundreds = ordinal_file("hundreds.tsv")
        space = self.DELETE_SPACE
        component = pynini.union(
            add_weight(digit, 0.4),
            add_weight(ties + (space + digit | insert("0")), 0.2),
            hundreds + (space + ties | insert("0")) + (space + digit | insert("0")),
        )
        # Leading ordinals have a negative weight so they are always attached.
        in_front = (add_weight(component, -0.1) + space).star
        graph = in_front + component
        exception = digit.copy().project("input")
        graph = (pynini.difference(graph.copy().project("input"), exception) @ graph).optimize()
        masculine = ((self.VSIGMA + "o") @ graph + insert("º")).optimize()
        feminine = ((self.VSIGMA + "a") @ graph + insert("ª")).optimize()
        self.graph = (masculine | feminine).optimize()
        self.tagger = self.add_tokens(self.tag_field("integer", self.graph))
        self.verbalizer = self.delete_tokens(self.verbalize_field("integer", self.graph))
