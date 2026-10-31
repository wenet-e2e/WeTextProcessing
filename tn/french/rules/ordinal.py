# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from tn.utils import get_abs_path


TOKEN_ORDER = {"ordinal": ["integer", "morphosyntactic_features"]}


class Ordinal(Processor):
    """``21e`` -> ``ordinal { integer: "21" morphosyntactic_features: "e" }`` -> ``vingt et unième``."""

    def __init__(self, cardinal):
        super().__init__("ordinal", token_orders=TOKEN_ORDER)
        self.suffixes = pynini.string_file(get_abs_path("french/data/ordinals/suffixes.tsv"))
        irregular = pynini.string_file(get_abs_path("french/data/ordinals/irregular_numbers.tsv"))

        # ``combine`` joins the spelled-out number with the spoken suffix. The
        # final word may change its root ("quatre" + "ième" -> "quatrième").
        words = self.NOT_QUOTE.plus
        prefix = (words + " ").star
        exceptions = (prefix + self.input_projection(irregular) | "un").optimize()
        spoken_suffixes = self.suffixes.copy().project("output").optimize()
        regular = pynini.difference(words, exceptions) + spoken_suffixes
        premier = pynini.cross("un", "prem") + spoken_suffixes
        root_change = prefix + irregular + delete("ième") + pynini.accep("s").ques
        self.combine = (regular | premier | root_change).optimize()

        def semantic(integer, suffix):
            return ((integer + suffix) @ self.combine).optimize()

        self.semantic = semantic
        raw = semantic(cardinal.graph, self.suffixes)
        tagger = self.tag_field("integer", cardinal.graph) + insert(" ")
        tagger += self.tag_field("morphosyntactic_features", self.suffixes)
        self.tagger = self.add_tokens(self.input_projection(raw) @ tagger)

        integer = self.verbalize_field("integer", cardinal.graph)
        suffix = self.verbalize_field("morphosyntactic_features", self.suffixes, leading_space=True)
        self.verbalizer = self.delete_tokens(semantic(integer, suffix))
