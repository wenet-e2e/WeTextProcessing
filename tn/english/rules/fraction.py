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

from pynini import accep, cdrewrite, cross, compose, string_file, union
from pynini.examples import plurals
from pynini.lib.pynutil import delete, insert

from tn.english.rules.cardinal import Cardinal
from tn.english.rules.ordinal import Ordinal
from tn.processor import Processor
from tn.utils import get_abs_path


class Fraction(Processor):

    def __init__(self, deterministic: bool = False):
        """
        Args:
            deterministic: if True will provide a single transduction option,
                for False multiple transduction are generated (used for audio-based normalization)
        """
        super().__init__("fraction", ordertype="en_tn")
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
        integer = insert('integer_part: "') + cardinal_graph + insert('"')
        numerator = (
            insert('numerator: "')
            + cardinal_graph
            + (cross("/", '" ') | cross(" / ", '" '))
        )

        endings = ["rd", "th", "st", "nd"]
        endings += [x.upper() for x in endings]
        optional_end = cross(union(*endings), "").ques

        denominator = (
            insert('denominator: "') + cardinal_graph + optional_end + insert('"')
        )

        graph = (integer + accep(" ")).ques + (numerator + denominator)
        graph |= (integer + (accep(" ") | insert(" "))).ques + compose(
            string_file(get_abs_path("english/data/number/fraction.tsv")),
            (numerator + denominator),
        )

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

        integer = delete('integer_part: "') + self.NOT_QUOTE.star + delete('" ')
        denominator_one = cross('denominator: "one"', "over one")
        denominator_half = cross('denominator: "two"', "half")
        denominator_quarter = cross('denominator: "four"', "quarter")

        denominator_rest = (
            delete('denominator: "') + self.NOT_QUOTE.star @ suffix + delete('"')
        )

        denominators = plurals._priority_union(
            denominator_one,
            plurals._priority_union(
                denominator_half,
                plurals._priority_union(
                    denominator_quarter, denominator_rest, self.VSIGMA
                ),
                self.VSIGMA,
            ),
            self.VSIGMA,
        ).optimize()
        if not self.deterministic:
            denominators |= (
                delete('denominator: "') + (accep("four") @ suffix) + delete('"')
            )

        numerator_one = delete('numerator: "') + accep("one") + delete('" ')
        numerator_one = numerator_one + insert(" ") + denominators
        numerator_rest = (
            delete('numerator: "') + (self.NOT_QUOTE.star - accep("one")) + delete('" ')
        )
        numerator_rest = numerator_rest + insert(" ") + denominators
        numerator_rest @= cdrewrite(
            plurals._priority_union(cross("half", "halves"), insert("s"), self.VSIGMA),
            "",
            "[EOS]",
            self.VSIGMA,
        )

        graph = numerator_one | numerator_rest

        conjunction = insert("and ")

        integer = (integer + insert(" ") + conjunction).ques

        graph = integer + graph
        graph @= cdrewrite(
            cross("and one half", "and a half") | cross("over ones", "over one"),
            "",
            "[EOS]",
            self.VSIGMA,
        )

        self.graph_v = graph
        delete_tokens = self.delete_tokens(self.graph_v)
        self.verbalizer = delete_tokens.optimize()
