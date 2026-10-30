# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pynini
from pynini.lib import pynutil

from tn.processor import Processor


TOKEN_ORDER = {}


class Fraction(Processor):
    """German numerator-before-denominator fractions with flat raw fields."""

    def __init__(self, cardinal, ordinal, deterministic=True):
        super().__init__("fraction", token_orders=TOKEN_ORDER)
        self.deterministic = deterministic
        change_one = pynini.cdrewrite(
            pynutil.add_weight(pynini.cross("eins", "ein"), -0.0001),
            "[BOS]", "[EOS]", self.VSIGMA,
        )
        self.integer_graph = (cardinal.graph @ change_one).optimize()
        self.numerator_graph = (cardinal.graph @ change_one).optimize()
        denominator_stem = (
            pynini.cdrewrite(pynini.closure(ordinal.stem, 0, 1), "", "[EOS]", self.VSIGMA)
            + pynutil.insert("tel")
        )
        change_two = pynini.cdrewrite(pynini.cross("zweitel", "halbe"), "[BOS]", "[EOS]", self.VSIGMA)
        self.denominator_graph = (cardinal.graph @ denominator_stem @ change_two).optimize()

        sign_tagger = pynini.closure(self.tag_field("negative", cardinal.sign) + pynutil.insert(" "), 0, 1)
        integer_tagger = pynini.closure(
            self.tag_field("integer_part", self.integer_graph) + pynini.accep(" "), 0, 1,
        )
        self.fields_tagger = (
            sign_tagger + integer_tagger + self.tag_field("numerator", self.numerator_graph)
            + pynutil.delete(pynini.union("/", " / ")) + pynutil.insert(" ")
            + self.tag_field("denominator", self.denominator_graph)
        ).optimize()

        sign_verbalizer = pynini.closure(
            self.verbalize_field("negative", cardinal.sign) + pynutil.delete(" "), 0, 1,
        )
        integer_verbalizer = pynini.closure(
            self.verbalize_field("integer_part", self.integer_graph) + pynutil.delete(" ")
            + pynutil.insert(" ") + pynini.closure(pynutil.insert("und ", weight=0.001), 0, 1),
            0, 1,
        )
        one_half = pynini.cdrewrite(pynini.cross("ein halbe", "ein halb"), "[BOS]", "[EOS]", self.VSIGMA)
        fraction_verbalizer = (
            self.verbalize_field("numerator", self.numerator_graph) + pynutil.insert(" ")
            + self.verbalize_field("denominator", self.denominator_graph, leading_space=True)
        ) @ one_half
        self.fields_verbalizer = (sign_verbalizer + integer_verbalizer + fraction_verbalizer).optimize()
        self.tagger = self.add_tokens(self.fields_tagger)
        self.verbalizer = self.delete_tokens(self.fields_verbalizer)
