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

from itn.hindi.rules.utils import SIGMA, string_file
from tn.processor import Processor

TOKEN_ORDER = {}

SUFFIXES = ("वाँ", "वां", "वें", "वे")
FEMININE_SUFFIXES = ("वीं", "वी")


class Ordinal(Processor):
    """``तेरहवां`` -> ``ordinal { integer: "तेरहवां" }`` -> ``१३वां``."""

    def __init__(self, cardinal):
        super().__init__("ordinal", ordertype="itn", token_orders=TOKEN_ORDER)
        masculine = SIGMA + pynini.union(string_file("ordinals/digit.tsv"), string_file("ordinals/teens_and_ties.tsv"),
                                         string_file("ordinals/hundred_digit.tsv"), *SUFFIXES)
        feminine = SIGMA + pynini.union(string_file("ordinals/digit_fem.tsv"),
                                        string_file("ordinals/teens_and_ties_fem.tsv"),
                                        string_file("ordinals/hundred_digit_fem.tsv"), *FEMININE_SUFFIXES)
        number = cardinal.graph_no_exception + pynini.union(*SUFFIXES, *FEMININE_SUFFIXES)
        self.graph = ((masculine | feminine) @ number).optimize()
        self.tagger = self.add_tokens(self.tag_field("integer", self.graph))
        self.verbalizer = self.delete_tokens(self.verbalize_field("integer", self.graph))
