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
from pynini.lib.pynutil import delete, insert

from itn.hindi.rules.utils import DELETE_EXTRA_SPACE, DELETE_SPACE, string_file
from tn.processor import Processor

TOKEN_ORDER = {}

POINT = "दशमलव"
NEGATIVE = "ऋण"


class Decimal(Processor):
    """``ऋण एक दशमलव दो छह लाख`` ->
    ``decimal { negative: "ऋण" integer_part: "एक" fractional_part: "दो छह" quantity: "लाख" }`` -> ``-१.२६ लाख``.
    """

    def __init__(self, cardinal):
        super().__init__("decimal", ordertype="itn", token_orders=TOKEN_ORDER)
        single = cardinal.digit | cardinal.zero
        self.fractional = ((single + DELETE_SPACE).star + single).optimize()
        self.integer = cardinal.graph_no_exception
        self.quantity = string_file("numbers/thousands.tsv")

        # Bare semantic graph (no sign), reused by measure and money: "एक दशमलव पाँच लाख" -> "१.५ लाख".
        number = (self.integer + DELETE_EXTRA_SPACE).ques + delete(POINT) + DELETE_EXTRA_SPACE + insert(".")
        number += self.fractional
        self.graph_wo_negative = (number + (DELETE_EXTRA_SPACE + insert(" ") + self.quantity).ques).optimize()

        def field(name, graph):
            return self.tag_field(name, graph) + DELETE_EXTRA_SPACE + insert(" ")

        tagger = field("negative", pynini.accep(NEGATIVE)).ques + field("integer_part", self.integer).ques
        tagger += delete(POINT) + DELETE_EXTRA_SPACE + self.tag_field("fractional_part", self.fractional)
        tagger += (DELETE_EXTRA_SPACE + insert(" ") + self.tag_field("quantity", self.quantity)).ques
        self.tagger = self.add_tokens(tagger)

        verbalizer = (self.verbalize_field("negative", pynini.cross(NEGATIVE, "-")) + delete(" ")).ques
        verbalizer += (self.verbalize_field("integer_part", self.integer) + delete(" ")).ques
        verbalizer += insert(".") + self.verbalize_field("fractional_part", self.fractional)
        verbalizer += (insert(" ") + self.verbalize_field("quantity", leading_space=True)).ques
        self.verbalizer = self.delete_tokens(verbalizer)
