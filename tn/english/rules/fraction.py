# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2024, WENET COMMUNITY.  Xingchen Song (sxc19@tsinghua.org.cn).
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
from pynini.examples import plurals
from pynini.lib import pynutil

from tn.english.rules.cardinal import Cardinal
from tn.english.rules.ordinal import Ordinal
from tn.processor import Processor
from tn.utils import get_abs_path


class Fraction(Processor):

    def __init__(self, deterministic: bool = False, cardinal=None, ordinal=None):
        super().__init__("fraction", ordertype="en_tn")
        self.deterministic = deterministic
        self.cardinal = cardinal or Cardinal(deterministic)
        self.ordinal = ordinal or Ordinal(deterministic, cardinal=self.cardinal)
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        """
        Finite state transducer for classifying fraction
        "23 4/5" ->
        fraction { integer_part: "23" numerator: "4" denominator: "5" }
        "23 4/5th" ->
        fraction { integer_part: "23" numerator: "4" denominator: "5th" }
        """
        cardinal_graph = self.cardinal.graph
        self.integer_graph = cardinal_graph
        integer = self.tag_field("integer_part", self.integer_graph)
        numerator = self.tag_field("numerator", cardinal_graph) + (pynini.cross("/", " ") | pynini.cross(" / ", " "))

        endings = ["rd", "th", "st", "nd"]
        endings += [x.upper() for x in endings]
        optional_end = pynini.cross(pynini.union(*endings), "").ques

        self.numerator_graph = cardinal_graph
        self.denominator_graph = cardinal_graph + optional_end
        denominator = self.tag_field("denominator", self.denominator_graph)

        graph = (integer + pynini.accep(" ")).ques + (numerator + denominator)
        self.symbol_graph = pynini.string_file(get_abs_path("english/data/number/fraction.tsv"))
        graph |= (integer + pynini.accep(" ")).ques + self.tag_field("value", self.symbol_graph)

        self.graph = graph
        final_graph = self.add_tokens(self.graph)
        self.tagger = final_graph.optimize()

    def build_verbalizer(self):
        """
        Finite state transducer for verbalizing fraction
            e.g. fraction { integer_part: "23" numerator: "4" denominator: "5" } ->
            twenty three and four fifth
        """
        suffix = self.ordinal.suffix

        integer = pynutil.delete('integer_part: "') + self.integer_graph + pynutil.delete('" ')
        denominator_value = pynutil.delete('denominator: "') + self.denominator_graph + pynutil.delete('"')
        denominator_one = denominator_value @ pynini.cross("one", "over one")
        denominator_half = denominator_value @ pynini.cross("two", "half")
        denominator_quarter = denominator_value @ pynini.cross("four", "quarter")
        denominator_rest = pynutil.add_weight(denominator_value @ suffix, 0.0001)

        denominators = plurals._priority_union(
            denominator_one,
            plurals._priority_union(
                denominator_half,
                plurals._priority_union(denominator_quarter, denominator_rest, self.VCHAR.star),
                self.VCHAR.star,
            ),
            self.VCHAR.star,
        ).optimize()
        if not self.deterministic:
            denominators |= pynutil.add_weight(
                denominator_value @ (pynini.accep("four") @ suffix),
                0.0001,
            )

        numerator_value = pynutil.delete('numerator: "') + self.numerator_graph + pynutil.delete('" ')
        numerator_one = numerator_value @ pynini.accep("one")
        numerator_one = numerator_one + self.INSERT_SPACE + denominators
        numerator_rest = numerator_value @ pynini.difference(self.NOT_QUOTE.star, pynini.accep("one"))
        numerator_rest = numerator_rest + self.INSERT_SPACE + denominators
        numerator_rest @= pynini.cdrewrite(
            plurals._priority_union(pynini.cross("half", "halves"), pynutil.insert("s"), self.VCHAR.star),
            "",
            "[EOS]",
            self.VCHAR.star,
        )

        graph = numerator_one | numerator_rest

        conjunction = pynutil.insert("and ")

        integer = (integer + self.INSERT_SPACE + conjunction).ques

        graph = integer + graph
        graph @= pynini.cdrewrite(
            pynini.cross("and one half", "and a half") | pynini.cross("over ones", "over one"),
            "",
            "[EOS]",
            self.VCHAR.star,
        )

        raw_numerator = (pynutil.insert('numerator: "') + self.input_projection(self.numerator_graph) +
                         (pynini.cross("/", '" ') | pynini.cross(" / ", '" ')))
        raw_denominator = (pynutil.insert('denominator: "') + self.input_projection(self.denominator_graph) +
                           pynutil.insert('"'))
        raw_integer = (pynini.accep('integer_part: "') + self.input_projection(self.integer_graph) + pynini.accep('" '))
        symbol_body = raw_integer.ques + (pynutil.delete('value: "') +
                                          (self.symbol_graph @ (raw_numerator + raw_denominator)) + pynutil.delete('"'))
        symbol = symbol_body @ graph
        graph |= symbol
        self.graph_v = graph
        delete_tokens = self.delete_tokens(self.graph_v)
        self.verbalizer = delete_tokens.optimize()
