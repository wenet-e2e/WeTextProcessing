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
from pynini.lib.pynutil import add_weight, delete, insert

from itn.french.rules.cardinal import load
from itn.french.rules.decimal import WHITE_SPACE
from tn.processor import Processor


TOKEN_ORDER = {"date": ["day", "month", "year"]}


class Date(Processor):
    """``premier janvier deux-mille-treize`` -> ``1ᵉʳ janvier 2013``; all three fields stay verbatim."""

    def __init__(self, cardinal):
        super().__init__("date", ordertype="itn", token_orders=TOKEN_ORDER)
        number = cardinal.graph_no_exception
        day = (number | pynini.cross("premier", "1")).optimize()
        month = load("months")
        separator = delete(WHITE_SPACE.plus) + insert(" ")

        tagger = self.tag_field("day", day) + separator + self.tag_field("month", month)
        tagger += (separator + self.tag_field("year", number)).ques
        self.tagger = self.add_tokens(tagger)

        # The first of the month is written as an ordinal.
        day = day @ (self.DIGIT.plus | add_weight(pynini.cross("1", "1ᵉʳ"), -1))
        verbalizer = self.verbalize_field("day", day) + pynini.accep(" ") + self.verbalize_field("month", month)
        verbalizer += (pynini.accep(" ") + self.verbalize_field("year", number)).ques
        self.verbalizer = self.delete_tokens(verbalizer)
