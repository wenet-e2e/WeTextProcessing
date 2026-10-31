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

from tn.processor import Processor
from tn.swedish.rules.utils import alt, rewrite, tsv

TOKEN_ORDER = {}


class Fraction(Processor):
    """Raw ``23 4/5`` or ``2 ½`` fractions; denominators come from ordinals."""

    def __init__(self, cardinal, ordinal, deterministic=True):
        super().__init__("fraction", token_orders=TOKEN_ORDER)
        self.deterministic = deterministic
        endings = pynini.string_map([
            ("ljarte", "ljarddel"), ("tionde", "tiondel"), ("tonde", "tondel"), ("ljonte", "ljondel"),
            ("lliarte", "lliarddel"), ("llionte", "lliondel"), ("tusende", "tusendel"), ("te", "tedel"),
            ("de", "dedel"), ("je", "jedel"), ("drade", "dradel"), ("a", "adel"),
        ])
        lexicalised = pynini.string_map([("andradel", "halv"), ("fjärdedel", "kvart")])
        fractions = rewrite(rewrite(ordinal.graph, endings, "", "[EOS]"), lexicalised, "[BOS]", "[EOS]")
        if not deterministic:
            alternative = rewrite(fractions, pynini.string_map([
                ("tondel", "tondedel"), ("tiondel", "tiondedel")]), "", "[EOS]")
            alternative = rewrite(alternative, pynini.string_map([
                ("halv", "andradel"), ("kvart", "fjärdedel"), ("kvart", "kvarts")]), "[BOS]", "[EOS]")
            fractions |= alt(alternative)
        self.fractions = fractions.optimize()
        plural = pynini.string_map([("kvart", "kvartar"), ("halv", "halva"), ("del", "delar")])
        self.fractions_plural = rewrite(self.fractions, plural, "", "[EOS]")

        numerator_one = pynutil.delete("1")
        if not deterministic:
            numerator_one |= alt(pynini.cross("1", "en "))
        numerator_rest = cardinal.graph_no_one_en
        # "N/D" text, e.g. the expansion of a vulgar fraction symbol.
        slash = pynutil.delete("/")
        self.text_graph = (
            numerator_one + slash + self.fractions
            | numerator_rest + pynini.cross("/", " ") + self.fractions_plural
        ).optimize()
        vulgar = (tsv("numbers/fraction") @ self.text_graph).optimize()

        numerator_t = self.tag_field("numerator", numerator_one | numerator_rest)
        numerator_t += pynutil.delete(pynini.union("/", " / "))
        denominator_t = pynutil.insert(" ") + self.tag_field("denominator", self.fractions)
        numerator_v = (
            self.verbalize_field("numerator", numerator_one) + pynutil.delete(" ")
            + self.verbalize_field("denominator", self.fractions)
            | self.verbalize_field("numerator", numerator_rest) + pynutil.insert(" ")
            + self.verbalize_field("denominator", self.fractions_plural, leading_space=True)
        )
        split_t = numerator_t + denominator_t
        vulgar_t = self.tag_field("fraction", vulgar)
        vulgar_v = self.verbalize_field("fraction", vulgar)

        integer_t = self.tag_field("integer_part", cardinal.graph)
        conjunction = pynutil.insert(" och ")
        if not deterministic:
            conjunction |= alt(pynutil.insert(" "))
        integer_v = self.verbalize_field("integer_part", cardinal.graph) + pynutil.delete(" ") + conjunction
        optional_space = pynini.accep(" ") | pynutil.insert(" ")
        self.fields_tagger = (
            (integer_t + pynini.accep(" ")).ques + split_t
            | (integer_t + optional_space).ques + vulgar_t
        ).optimize()
        self.fields_verbalizer = (integer_v.ques + (numerator_v | vulgar_v)).optimize()
        self.tagger = self.add_tokens(self.fields_tagger)
        self.verbalizer = self.delete_tokens(self.fields_verbalizer)
