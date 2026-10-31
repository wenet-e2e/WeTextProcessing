# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from tn.utils import get_abs_path, load_labels


TOKEN_ORDER = {}


class Fraction(Processor):
    """Raw numerators and denominators; the denominator decides the reading.

    Denominators 2-10, 100 and 1000 are ordinals (plural unless the
    numerator is 1), all others are cardinals followed by "avos".
    """

    def __init__(self, cardinal, ordinal):
        super().__init__("fraction", token_orders=TOKEN_ORDER)
        specials = {row[0]: row[1] for row in load_labels(get_abs_path("portuguese/data/fractions/specials.tsv"))}
        exceptions = dict(load_labels(get_abs_path("portuguese/data/fractions/ordinal_exceptions.tsv")))
        powers = dict(load_labels(get_abs_path("portuguese/data/fractions/powers_of_ten.tsv")))
        connector = " " + specials["connector"].strip() + " "
        plural = specials["plural_suffix"]
        avos = " " + specials["avos_suffix"].strip()

        ordinal_words = {}
        for value in list(range(2, 11)) + [100]:
            word = pynini.shortestpath(pynini.accep(str(value)) @ ordinal.masculine).string()
            ordinal_words[str(value)] = exceptions.get(word, word)
        ordinal_words["1000"] = powers[cardinal.thousand]
        ordinal_denominator = pynini.string_map(ordinal_words.items())
        cardinal_denominator = pynini.difference(self.DIGIT.plus, pynini.union(*ordinal_words)) @ cardinal.graph
        self.sign = cardinal.sign

        one = pynini.accep("1") @ cardinal.graph
        other = pynini.difference(self.VSIGMA, "1") @ cardinal.graph
        slash = pynini.union("/", "\u2044", "\u2215")
        slash = delete(slash | " " + slash + " ") + insert(" ")

        sign_t = (self.tag_field("negative", self.sign) + insert(" ")).ques
        integer_t = self.tag_field("integer_part", cardinal.graph) + delete(self.SPACE.plus) + insert(" ")
        fraction_t = (
            self.tag_field("numerator", cardinal.graph) + slash
            + self.tag_field("denominator", ordinal_denominator | cardinal_denominator)
        )
        self.fields_tagger = (sign_t + integer_t.ques + fraction_t).optimize()

        def numerator(graph):
            return self.verbalize_field("numerator", graph) + insert(" ")

        def denominator(graph):
            return self.verbalize_field("denominator", graph, leading_space=True)

        sign_v = (self.verbalize_field("negative", self.sign) + delete(" ")).ques
        integer_v = self.verbalize_field("integer_part", cardinal.graph) + delete(" ") + insert(connector)
        fraction_v = (
            numerator(one) + denominator(ordinal_denominator)
            | numerator(other) + denominator(ordinal_denominator) + insert(plural)
            | numerator(cardinal.graph) + denominator(cardinal_denominator) + insert(avos)
        )
        self.fields_verbalizer = (sign_v + integer_v.ques + fraction_v).optimize()
        self.tagger = self.add_tokens(self.fields_tagger)
        self.verbalizer = self.delete_tokens(self.fields_verbalizer)
