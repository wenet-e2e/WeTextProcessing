# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


TOKEN_ORDER = {"fraction": ["negative", "integer_part", "numerator", "denominator"]}


class Fraction(Processor):
    """``-12 1/4`` keeps its parts raw and reads as "âm mười hai và một phần tư"."""

    def __init__(self, cardinal):
        super().__init__("fraction", token_orders=TOKEN_ORDER)
        exceptions = pynini.string_file(get_abs_path("vietnamese/data/fraction/denominator_exceptions.tsv"))
        number = self.DIGIT.plus
        self.number_graph = (number @ cardinal.graph).optimize()
        self.number_raw = (number @ cardinal.raw).optimize()
        # The exception table ("4" -> "tư") replaces the plain cardinal reading.
        regular = pynini.difference(number, self.input_projection(exceptions))
        self.denominator_graph = (exceptions | regular @ self.number_graph).optimize()

        sep = insert(" ")
        simple = self.tag_field("numerator", self.number_raw) + delete("/") + sep
        simple += self.tag_field("denominator", self.number_raw)
        mixed = self.tag_field("integer_part", self.number_raw) + delete(" ") + sep + simple
        self.fields_tagger = (simple | mixed).optimize()
        self.tagger = self.add_tokens(cardinal.negative_tagger.ques + self.fields_tagger)

        simple_v = self.verbalize_field("numerator", self.number_graph) + insert(" phần ")
        simple_v += self.verbalize_field("denominator", self.denominator_graph, leading_space=True)
        mixed_v = self.verbalize_field("integer_part", self.number_graph) + insert(" và ") + delete(" ") + simple_v
        self.fields_verbalizer = (simple_v | mixed_v).optimize()
        self.verbalizer = self.delete_tokens(cardinal.negative_verbalizer.ques + self.fields_verbalizer)
