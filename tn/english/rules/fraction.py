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

from tn.processor import Processor
from tn.utils import get_abs_path
from tn.english.rules.cardinal import Cardinal
from tn.english.rules.ordinal import Ordinal


class Fraction(Processor):

    def __init__(self, deterministic: bool = False):
        """
        Args:
            deterministic: if True will provide a single transduction option,
                for False multiple transduction are generated (used for audio-based normalization)
        """
        super().__init__('fraction', ordertype="en_tn")
        self.deterministic = deterministic
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        """
        Finite state transducer for classifying fraction
        "23 4/5" ->
        fraction { integer_part: "twenty three" numerator: "four" denominator: "five" }
        "23 4/5th" ->
        fraction { integer_part: "twenty three" numerator: "four" denominator: "five" }
        """
        cardinal_graph = Cardinal(self.deterministic).graph
        integer = pynutil.insert(
            "integer_part: \"") + cardinal_graph + pynutil.insert("\"")
        numerator = (pynutil.insert("numerator: \"") + cardinal_graph +
                     (pynini.cross("/", "\" ") | pynini.cross(" / ", "\" ")))

        endings = ["rd", "th", "st", "nd"]
        endings += [x.upper() for x in endings]
        optional_end = pynini.cross(pynini.union(*endings), "").ques

        denominator = pynutil.insert(
            "denominator: \""
        ) + cardinal_graph + optional_end + pynutil.insert("\"")

        graph = (integer + pynini.accep(" ")).ques + (numerator + denominator)
        graph |= (
            integer + (pynini.accep(" ") | pynutil.insert(" "))).ques + pynini.compose(
                pynini.string_file(
                    get_abs_path("english/data/number/fraction.tsv")),
                (numerator + denominator))

        self.graph = graph
        final_graph = self.add_tokens(self.graph)
        self.tagger = final_graph.optimize()

    def build_verbalizer(self):
        """
        Finite state transducer for verbalizing fraction
            e.g. fraction { integer_part: "twenty three" numerator: "four" denominator: "five" } ->
            twenty three and four fifth
        """
        suffix = Ordinal(self.deterministic).suffix

        integer = pynutil.delete("integer_part: \"") + self.NOT_QUOTE.star + pynutil.delete("\" ")
        denominator_one = pynini.cross("denominator: \"one\"", "over one")
        denominator_half = pynini.cross("denominator: \"two\"", "half")
        denominator_quarter = pynini.cross("denominator: \"four\"", "quarter")

        denominator_rest = (pynutil.delete("denominator: \"") +
                            self.NOT_QUOTE.star @ suffix +
                            pynutil.delete("\""))

        denominators = plurals._priority_union(
            denominator_one,
            plurals._priority_union(
                denominator_half,
                plurals._priority_union(denominator_quarter, denominator_rest,
                                        self.VCHAR.star),
                self.VCHAR.star,
            ),
            self.VCHAR.star,
        ).optimize()
        if not self.deterministic:
            denominators |= pynutil.delete("denominator: \"") + (
                pynini.accep("four") @ suffix) + pynutil.delete("\"")

        numerator_one = pynutil.delete("numerator: \"") + pynini.accep(
            "one") + pynutil.delete("\" ")
        numerator_one = numerator_one + self.INSERT_SPACE + denominators
        numerator_rest = (
            pynutil.delete("numerator: \"") +
            (self.NOT_QUOTE.star - pynini.accep("one")) +
            pynutil.delete("\" "))
        numerator_rest = numerator_rest + self.INSERT_SPACE + denominators
        numerator_rest @= pynini.cdrewrite(
            plurals._priority_union(pynini.cross("half", "halves"),
                                    pynutil.insert("s"),
                                    self.VCHAR.star),
            "",
            "[EOS]",
            self.VCHAR.star,
        )

        graph = numerator_one | numerator_rest

        conjunction = pynutil.insert("and ")

        integer = (integer + self.INSERT_SPACE + conjunction).ques

        graph = integer + graph
        graph @= pynini.cdrewrite(
            pynini.cross("and one half", "and a half")
            | pynini.cross("over ones", "over one"), "", "[EOS]",
            self.VCHAR.star)

        self.graph_v = graph
        delete_tokens = self.delete_tokens(self.graph_v)
        self.verbalizer = delete_tokens.optimize()
