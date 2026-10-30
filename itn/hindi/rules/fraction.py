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

from itn.hindi.rules.utils import DEDH, DELETE_EXTRA_SPACE, DELETE_SPACE
from tn.processor import Processor

TOKEN_ORDER = {}


class Fraction(Processor):
    """``एक सही एक बटा दो`` -> ``fraction { value: "एक सही एक बटा दो" }`` -> ``१ १/२``.

    Also covers the quarter words: ``सवा पैंतीस`` -> ``३५ १/४``, ``पौना`` -> ``३/४``, ``दो तिहाई`` -> ``२/३``.
    """

    def __init__(self, cardinal):
        super().__init__("fraction", ordertype="itn", token_orders=TOKEN_ORDER)
        number = cardinal.graph_no_exception
        integer = number + insert(" ") + DELETE_SPACE
        bata = delete(" बटा ") | delete(" बटे ")
        fraction = number + bata + insert("/") + number
        mixed = integer + DELETE_EXTRA_SPACE + delete("सही") + DELETE_SPACE + fraction

        def quarter(prefix, value, weight):
            return add_weight(delete(prefix) + DELETE_SPACE + integer + insert(value) + DELETE_SPACE, weight)

        def fixed(prefix, value, weight=None):
            graph = delete(prefix) + DELETE_SPACE + insert(value) + DELETE_SPACE
            return graph if weight is None else add_weight(graph, weight)

        quarters = (quarter("साढ़े", "१/२", -0.01) | quarter("सवा", "१/४", -0.001) | quarter("पौने", "३/४", -0.01)
                    | fixed(DEDH, "१ १/२", -0.01) | fixed("ढाई", "२ १/२", -0.1)
                    | fixed(pynini.union("आधा", "साढ़े"), "१/२") | fixed("सवा", "१/४")
                    | fixed(pynini.union("पौन", "पौना"), "३/४")
                    | number + DELETE_SPACE + delete("तिहाई") + DELETE_SPACE + insert("/३")
                    | number + DELETE_SPACE + delete("चौथाई") + DELETE_SPACE + insert("/४"))
        graph = fraction | mixed | quarters
        self.graph = ((pynini.cross("ऋण", "-") + DELETE_EXTRA_SPACE).ques + graph).optimize()
        self.tagger = self.add_tokens(self.tag_field("value", self.graph))
        self.verbalizer = self.delete_tokens(self.verbalize_field("value", self.graph))
