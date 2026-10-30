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

from itn.hindi.rules.utils import DEDH, DELETE_SPACE, devanagari_numbers, spoken_to_written
from tn.processor import Processor

TOKEN_ORDER = {}


class Time(Processor):
    """``दो बजकर ग्यारह मिनट`` -> ``time { value: "दो बजकर ग्यारह मिनट" }`` -> ``२:११``."""

    def __init__(self, cardinal):
        super().__init__("time", ordertype="itn", token_orders=TOKEN_ORDER)
        hour = (cardinal.digit | cardinal.teens_and_ties) @ devanagari_numbers(1, 25)
        paune_hour = spoken_to_written("time/hour_for_paune.tsv")
        minute = cardinal.single_digit_with_zero | cardinal.teens_and_ties
        baje = delete(pynini.union("बजके", "बजकर", "बजे", "घंटा"))
        minutes = DELETE_SPACE + delete("मिनट")
        seconds = DELETE_SPACE + delete("सेकंड")
        colon = DELETE_SPACE + insert(":")

        hms = hour + DELETE_SPACE + baje + colon + minute + minutes + colon + minute + seconds
        hm = add_weight(hour + DELETE_SPACE + baje.ques + colon + minute + minutes.ques, 0.01)
        hs = add_weight(hour + DELETE_SPACE + baje + colon + insert("००:") + minute + seconds, 0.01)
        ms = insert("००:") + minute + minutes + colon + minute + seconds
        hour_only = hour + DELETE_SPACE + baje + insert(":००")

        def quarter(prefix, hours, value):
            graph = delete(prefix) + DELETE_SPACE + hours + insert(value) + DELETE_SPACE + baje.star
            return add_weight(graph, 0.01) + (DELETE_SPACE + baje).star

        quarters = (add_weight(delete(DEDH) + DELETE_SPACE + baje + insert("१:३०") + DELETE_SPACE, 0.01)
                    | add_weight(delete("ढाई") + DELETE_SPACE + baje + insert("२:३०") + DELETE_SPACE, 0.01)
                    | quarter("साढ़े", hour, ":३०") | quarter("सवा", hour, ":१५") | quarter("पौने", paune_hour, ":४५"))
        self.graph = (hms | hm | hs | ms | hour_only | quarters).optimize()
        self.tagger = self.add_tokens(self.tag_field("value", self.graph))
        self.verbalizer = self.delete_tokens(self.verbalize_field("value", self.graph))
