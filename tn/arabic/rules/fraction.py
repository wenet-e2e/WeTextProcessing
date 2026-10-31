# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from tn.processor import Processor
from tn.utils import get_abs_path


class Fraction(Processor):
    """Fractions, e.g. ``1 1/2`` -> ``واحد و نصف`` and ``9/3`` -> ``تسعة أثلاث``.

    The tagger keeps the written integer, numerator and denominator (or the
    vulgar fraction character); the verbalizer picks the singular, dual or
    plural fraction noun for denominators up to ten, else ``X على Y``.
    """

    def __init__(self, cardinal):
        super().__init__("fraction", token_orders={})
        self.cardinal = cardinal
        self.build_graphs()
        self.build_tagger()
        self.build_verbalizer()

    def build_graphs(self):
        number = self.cardinal.graph
        one_to_ten = pynini.union(*[str(n) for n in range(1, 11)])
        singular = pynini.string_file(get_abs_path("arabic/data/number/fraction_singular.tsv"))
        dual = pynini.string_file(get_abs_path("arabic/data/number/fraction_dual.tsv"))
        plural = pynini.string_file(get_abs_path("arabic/data/number/fraction_plural.tsv"))
        three_to_ten = pynini.union(*[str(n) for n in range(3, 11)])
        # (numerator, denominator) readings; the generic one is a fallback.
        self.readings = (
            (delete("1"), one_to_ten @ singular),
            (delete("2"), one_to_ten @ dual),
            (three_to_ten @ number + insert(" "), one_to_ten @ plural),
            (add_weight(number + insert(" على "), 0.001), number),
        )
        self.number = number
        slash = pynini.union(*[numerator + delete("/") + denominator for numerator, denominator in self.readings])
        vulgar = pynini.string_file(get_abs_path("arabic/data/number/fraction.tsv"))
        self.vulgar = (vulgar @ slash).optimize()
        self.integer_graph = (number + insert(" و ")).optimize()

    def tag_number(self):
        """Raw fraction fields, shared with measure."""

        integer = self.tag_field("integer_part", self.integer_graph)
        slash = self.tag_field("numerator", self.number) + delete(pynini.union("/", " / ")) + insert(" ")
        slash += self.tag_field("denominator", self.number)
        vulgar = self.tag_field("vulgar", self.vulgar)
        return (integer + pynini.accep(" ")).ques + slash | (integer + delete(" ").ques + insert(" ")).ques + vulgar

    def verbalize_number(self):
        integer = self.verbalize_field("integer_part", self.integer_graph) + delete(" ")
        slash = pynini.union(*[
            self.verbalize_field("numerator", numerator) + self.verbalize_field("denominator", denominator, True)
            for numerator, denominator in self.readings
        ])
        return integer.ques + (slash | self.verbalize_field("vulgar", self.vulgar))

    def build_tagger(self):
        self.tagger = self.add_tokens(self.tag_number())

    def build_verbalizer(self):
        self.verbalizer = self.delete_tokens(self.verbalize_number())
