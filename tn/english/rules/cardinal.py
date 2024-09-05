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


class Cardinal(Processor):

    def __init__(self, deterministic: bool = False):
        """
        Args:
            deterministic: if True will provide a single transduction option,
                for False multiple transduction are generated (used for audio-based normalization)
        """
        super().__init__('cardinal', ordertype="en_tn")
        self.deterministic = deterministic
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        """
        Finite state transducer for classifying cardinals, e.g.
            -23 -> cardinal { negative: "true" integer: "twenty three" }
        """
        # TODO replace to have "oh" as a default for "0"
        graph = pynini.Far(
            get_abs_path(
                "english/data/number/cardinal_number_name.far")).get_fst()
        graph_au = pynini.Far(
            get_abs_path(
                "english/data/number/cardinal_number_name_au.far")).get_fst()
        self.graph_hundred_component_at_least_one_none_zero_digit = (
            pynini.closure(self.DIGIT, 2, 3)
            | pynini.difference(self.DIGIT, pynini.accep("0"))) @ graph

        graph_digit = pynini.string_file(
            get_abs_path("english/data/number/digit.tsv"))
        graph_zero = pynini.string_file(
            get_abs_path("english/data/number/zero.tsv"))

        single_digits_graph = pynini.invert(graph_digit | graph_zero)
        self.single_digits_graph = single_digits_graph + (
            self.INSERT_SPACE + single_digits_graph).star

        if not self.deterministic:
            # for a single token allow only the same normalization
            # "007" -> {"oh oh seven", "zero zero seven"} not {"oh zero seven"}
            single_digits_graph_zero = pynini.invert(graph_digit | graph_zero)
            single_digits_graph_oh = pynini.invert(graph_digit) | pynini.cross(
                "0", "oh")

            self.single_digits_graph = single_digits_graph_zero + (
                self.INSERT_SPACE + single_digits_graph_zero).star
            self.single_digits_graph |= single_digits_graph_oh + (
                self.INSERT_SPACE + single_digits_graph_oh).star

            single_digits_graph_with_commas = pynini.closure(
                self.single_digits_graph + self.INSERT_SPACE, 1,
                3) + (
                    pynutil.delete(",") + single_digits_graph +
                    self.INSERT_SPACE + single_digits_graph +
                    self.INSERT_SPACE + single_digits_graph).plus

        graph = (pynini.closure(self.DIGIT, 1, 3) +
                 ((pynutil.delete(",") + self.DIGIT**3).star
                  | (self.DIGIT**3).star)) @ graph

        self.graph = graph
        self.graph_with_and = self.add_optional_and(graph)

        if self.deterministic:
            long_numbers = pynini.compose(self.DIGIT**(5, ...),
                                          self.single_digits_graph).optimize()
            self.long_numbers = plurals._priority_union(
                long_numbers, self.graph_with_and,
                self.VCHAR.star).optimize()
            cardinal_with_leading_zeros = pynini.compose(
                pynini.accep("0") + self.DIGIT.star,
                self.single_digits_graph)
            final_graph = self.long_numbers | cardinal_with_leading_zeros
            final_graph |= self.add_optional_and(graph_au)
        else:
            leading_zeros = pynini.compose(
                pynini.accep("0").plus, self.single_digits_graph)
            cardinal_with_leading_zeros = (
                leading_zeros + self.INSERT_SPACE + pynini.compose(
                    self.DIGIT.star, self.graph_with_and))
            self.long_numbers = self.graph_with_and | pynutil.add_weight(
                self.single_digits_graph, 0.0001)
            # add small weight to non-default graphs to make sure the deterministic option is listed first
            final_graph = (self.long_numbers
                           | pynutil.add_weight(
                               single_digits_graph_with_commas, 0.0001)
                           | cardinal_with_leading_zeros).optimize()

            one_to_a_replacement_graph = (
                pynini.cross("one hundred", "a hundred")
                | pynini.cross("one thousand", "thousand")
                | pynini.cross("one million", "a million"))
            final_graph |= pynini.compose(
                final_graph,
                one_to_a_replacement_graph.optimize() +
                self.VCHAR.star).optimize()
            # remove commas for 4 digits numbers
            four_digit_comma_graph = (
                self.DIGIT - "0") + pynutil.delete(",") + self.DIGIT**3
            final_graph |= pynini.compose(four_digit_comma_graph.optimize(),
                                          final_graph).optimize()

        self.final_graph = final_graph
        optional_minus_graph = (
            pynutil.insert("negative: ") + pynini.cross("-", "\"true\" ")).ques
        final_graph = optional_minus_graph + pynutil.insert(
            "integer: \"") + final_graph + pynutil.insert("\"")
        final_graph = self.add_tokens(final_graph)
        self.tagger = final_graph.optimize()

    def add_optional_and(self, graph):
        graph_with_and = graph

        graph_with_and = pynutil.add_weight(graph, 0.00001)
        not_quote = self.NOT_QUOTE.star
        no_thousand_million = pynini.difference(
            not_quote, not_quote + pynini.union("thousand", "million") +
            not_quote).optimize()
        integer = (not_quote + pynutil.add_weight(
            pynini.cross("hundred ", "hundred and ") + no_thousand_million,
            -0.0001)).optimize()

        no_hundred = pynini.difference(self.VCHAR.star,
            not_quote + pynini.accep("hundred") + not_quote).optimize()
        integer |= (not_quote + pynutil.add_weight(
            pynini.cross("thousand ", "thousand and ") + no_hundred,
            -0.0001)).optimize()

        optional_hundred = pynini.compose((self.DIGIT - "0")**3,
                                          graph).optimize()
        optional_hundred = pynini.compose(
            optional_hundred,
            self.VCHAR.star + pynini.cross(" hundred", "") +
            self.VCHAR.star)
        graph_with_and |= pynini.compose(graph, integer).optimize()
        graph_with_and |= optional_hundred
        return graph_with_and

    def build_verbalizer(self):
        """
        Finite state transducer for verbalizing cardinal, e.g.
            cardinal { negative: "true" integer: "23" } -> minus twenty three
        """
        optional_sign = pynini.cross("negative: \"true\"", "minus ")
        if not self.deterministic:
            optional_sign |= pynini.cross("negative: \"true\"", "negative ")
            optional_sign |= pynini.cross("negative: \"true\"", "dash ")

        self.optional_sign = (optional_sign + self.DELETE_SPACE).ques

        integer = self.NOT_QUOTE.star

        self.integer = self.DELETE_SPACE + pynutil.delete(
            "\"") + integer + pynutil.delete("\"")
        integer = pynutil.delete("integer:") + self.integer

        self.numbers = self.optional_sign + integer
        delete_tokens = self.delete_tokens(self.numbers)
        self.verbalizer = delete_tokens.optimize()
