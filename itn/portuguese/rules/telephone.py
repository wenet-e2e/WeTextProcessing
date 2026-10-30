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
from pynini.lib.pynutil import delete, insert

from tn.processor import Processor


TOKEN_ORDER = {}


class Telephone(Processor):
    """Raw spoken phone numbers, grouped by length when verbalized.

    Digits are spoken one by one or in pairs ("meia" is six): 11 digits are
    written (12) 34567-8901, 10 as (12) 3456-7890, 9 as 12345-6789 and 8 as
    1234-5678.
    """

    def __init__(self, cardinal):
        super().__init__("telephone", ordertype="itn", token_orders=TOKEN_ORDER)
        space = self.DELETE_SPACE
        single = (cardinal.digit | pynini.cross("meia", "6") | cardinal.zero).optimize()
        double = pynini.union(
            cardinal.teen | cardinal.twenties,
            cardinal.ties + insert("0"),
            cardinal.ties + space + delete("e") + space + cardinal.digit,
            single + space + single,
        ).optimize()
        sep = delete(" ")

        def pairs(count):
            return double + (sep + double) ** (count - 1)

        area = insert("(") + double + sep + insert(") ")
        self.graph = pynini.union(
            area + single + sep + pairs(2) + sep + insert("-") + pairs(2),
            area + pairs(2) + sep + insert("-") + pairs(2),
            single + sep + pairs(2) + sep + insert("-") + pairs(2),
            pairs(2) + sep + insert("-") + pairs(2),
        ).optimize()
        self.tagger = self.add_tokens(self.tag_field("number_part", self.graph))
        self.verbalizer = self.delete_tokens(self.verbalize_field("number_part", self.graph))
