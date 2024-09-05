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

from pynini import (
    accep,
    cross,
    closure,
    compose,
    difference,
    Far,
    invert,
    string_file,
    union,
)
from pynini.examples import plurals
from pynini.lib.pynutil import add_weight, delete, insert

from tn.processor import Processor
from tn.utils import get_abs_path


class Cardinal(Processor):

    def __init__(self, deterministic: bool = False):
        """
        Args:
            deterministic: if True will provide a single transduction option,
                for False multiple transduction are generated (used for audio-based normalization)
        """
        super().__init__("cardinal", ordertype="en_tn")
        self.deterministic = deterministic
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        """
        Finite state transducer for classifying cardinals, e.g.
            -23 -> cardinal { negative: "true" integer: "twenty three" }
        """
        digit = invert(string_file(get_abs_path("english/data/number/digit.tsv")))
        zero = invert(string_file(get_abs_path("english/data/number/zero.tsv")))

        digits = digit | zero
        self.single_digits = digits + (insert(" ") + digits).star
        if not self.deterministic:
            # for a single token allow only the same normalization
            # "007" -> {"oh oh seven", "zero zero seven"} not {"oh zero seven"}
            single_digits_oh = digit | cross("0", "oh")
            self.single_digits |= (
                single_digits_oh + (insert(" ") + single_digits_oh).star
            )

        # TODO replace to have "oh" as a default for "0"
        graph = Far(
            get_abs_path("english/data/number/cardinal_number_name.far")
        ).get_fst()
        graph_au = Far(
            get_abs_path("english/data/number/cardinal_number_name_au.far")
        ).get_fst()
        self.graph_hundred_component_at_least_one_none_zero_digit = (
            closure(self.DIGIT, 2, 3) @ graph | digit
        )

        graph = (
            closure(self.DIGIT, 1, 3)
            + ((delete(",") + self.DIGIT**3).star | (self.DIGIT**3).star)
        ) @ graph

        self.graph = graph
        self.graph_with_and = self.add_optional_and(graph)

        if self.deterministic:
            long_numbers = (closure(self.DIGIT, 5) @ self.single_digits).optimize()
            self.long_numbers = plurals._priority_union(
                long_numbers, self.graph_with_and, self.VSIGMA
            ).optimize()
            cardinal_with_leading_zeros = compose(
                accep("0") + self.DIGIT.star, self.single_digits
            )
            final_graph = self.long_numbers | cardinal_with_leading_zeros
            final_graph |= self.add_optional_and(graph_au)
        else:
            leading_zeros = accep("0").plus @ self.single_digits
            cardinal_with_leading_zeros = (
                leading_zeros
                + insert(" ")
                + compose(self.DIGIT.star, self.graph_with_and)
            )
            self.long_numbers = self.graph_with_and | add_weight(
                self.single_digits, 0.0001
            )
            # add small weight to non-default graphs to make sure the deterministic option is listed first
            final_graph = (self.long_numbers | cardinal_with_leading_zeros).optimize()

            one_to_a_replacement_graph = (
                cross("one hundred", "a hundred")
                | cross("one thousand", "thousand")
                | cross("one million", "a million")
            ).optimize()
            final_graph |= compose(
                final_graph, one_to_a_replacement_graph + self.VSIGMA
            ).optimize()
            # remove commas for 4 digits numbers
            four_digit_comma_graph = (self.DIGIT - "0") + delete(",") + self.DIGIT**3
            final_graph |= compose(
                four_digit_comma_graph.optimize(), final_graph
            ).optimize()

        self.final_graph = final_graph
        optional_minus_graph = cross("-", 'negative: "true" ').ques
        final_graph = (
            optional_minus_graph + insert('integer: "') + final_graph + insert('"')
        )
        final_graph = self.add_tokens(final_graph)
        self.tagger = final_graph.optimize()

    def add_optional_and(self, graph):
        graph_with_and = graph

        graph_with_and = add_weight(graph, 0.00001)
        not_quote = self.NOT_QUOTE.star
        no_thousand_million = difference(
            not_quote, not_quote + union("thousand", "million") + not_quote
        ).optimize()
        integer = (
            not_quote
            + add_weight(
                cross("hundred ", "hundred and ") + no_thousand_million, -0.0001
            )
        ).optimize()

        no_hundred = difference(
            self.VSIGMA, not_quote + accep("hundred") + not_quote
        ).optimize()
        integer |= (
            not_quote
            + add_weight(cross("thousand ", "thousand and ") + no_hundred, -0.0001)
        ).optimize()

        optional_hundred = compose((self.DIGIT - "0") ** 3, graph).optimize()
        optional_hundred = compose(
            optional_hundred,
            self.VSIGMA + cross(" hundred", "") + self.VSIGMA,
        )
        graph_with_and |= compose(graph, integer).optimize()
        graph_with_and |= optional_hundred
        return graph_with_and

    def build_verbalizer(self):
        """
        Finite state transducer for verbalizing cardinal, e.g.
            cardinal { negative: "true" integer: "23" } -> minus twenty three
        """
        optional_sign = cross('negative: "true"', "minus ")
        if not self.deterministic:
            optional_sign |= cross('negative: "true"', "negative ")
            optional_sign |= cross('negative: "true"', "dash ")

        self.optional_sign = (optional_sign + self.DELETE_SPACE).ques

        integer = self.NOT_QUOTE.star
        self.integer = self.DELETE_SPACE + delete('"') + integer + delete('"')
        integer = delete("integer:") + self.integer

        self.numbers = self.optional_sign + integer
        delete_tokens = self.delete_tokens(self.numbers)
        self.verbalizer = delete_tokens.optimize()
