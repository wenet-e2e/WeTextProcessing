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
from pynini.lib import pynutil

from tn.english.rules.cardinal import Cardinal
from tn.english.rules.decimal import Decimal
from tn.english.rules.measure import SINGULAR_TO_PLURAL
from tn.processor import Processor
from tn.utils import get_abs_path, load_labels

maj_singular = pynini.string_file((get_abs_path("english/data/money/currency_major.tsv")))


class Money(Processor):

    def __init__(self, deterministic: bool = False, cardinal=None, decimal=None):
        super().__init__("money", ordertype="en_tn")
        self.deterministic = deterministic
        self.cardinal = cardinal or Cardinal(deterministic)
        self.decimal = decimal or Decimal(deterministic, cardinal=self.cardinal)
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        """
        Finite state transducer for classifying money, suppletive aware, e.g.
            $12.05 -> money { currency_maj: "$" integer_part: "12" fractional_part: "05" }
            $12.0500 -> money { currency_maj: "$" integer_part: "12" fractional_part: "0500" }
            $1 -> money { currency_maj: "$" integer_part: "1" }
            $1.00 -> money { currency_maj: "$" integer_part: "1" fractional_part: "00" }
            $0.05 -> money { currency_maj: "$" integer_part: "0" fractional_part: "05" }
            $1 million -> money { currency_maj: "$" integer_part: "1" quantity: "million" }
            $1.2 million -> money { currency_maj: "$" integer_part: "1" fractional_part: "2" quantity: "million" }
            $1.2320 -> money { currency_maj: "$" integer_part: "1" fractional_part: "2320" }
        """
        cardinal = self.cardinal
        decimal = self.decimal
        cardinal_graph = cardinal.graph_with_and
        graph_decimal_final = decimal.final_graph_wo_negative_w_abbr

        maj_singular_labels = load_labels(get_abs_path("english/data/money/currency_major.tsv"))
        maj_unit_plural = maj_singular @ SINGULAR_TO_PLURAL
        maj_unit_singular = maj_singular

        self.maj_unit_singular = maj_unit_singular
        self.maj_unit_plural = maj_unit_plural
        graph_maj_singular = self.tag_field("currency_maj", self.maj_unit_singular)
        graph_maj_plural = self.tag_field("currency_maj", self.maj_unit_plural)

        self.integer_one_graph = pynini.cross("1", "one")
        graph_integer_one = self.tag_field("integer_part", self.integer_one_graph)
        decimal_nonzero = ((self.DIGIT | pynini.accep(",")).star + pynini.accep(".") + self.DIGIT.star + (self.DIGIT - "0") +
                           self.DIGIT.star)
        decimal_with_quantity = self.VCHAR.star + self.ALPHA

        graph_decimal = (graph_maj_plural + self.INSERT_SPACE +
                         (decimal_nonzero | decimal_with_quantity) @ graph_decimal_final)

        self.integer_other_graph = (self.VCHAR.star - "1") @ cardinal_graph
        graph_integer = self.tag_field("integer_part", self.integer_other_graph)

        graph_integer_only = graph_maj_singular + self.INSERT_SPACE + graph_integer_one
        graph_integer_only |= graph_maj_plural + self.INSERT_SPACE + graph_integer

        self.zero_fraction_graph = pynutil.delete("0").plus
        optional_zero_fraction = (pynutil.delete(".") + self.INSERT_SPACE +
                                  self.tag_field("fractional_part", self.zero_fraction_graph)).ques

        final_graph = (graph_integer_only + optional_zero_fraction) | graph_decimal

        self.tagger = self.add_tokens(final_graph.optimize())

    def build_verbalizer(self):
        """
        Finite state transducer for verbalizing money, e.g.
            money { currency_maj: "$" integer_part: "12" fractional_part: "05" } -> twelve point oh five dollars
        """
        decimal = self.decimal
        keep_space = pynini.accep(" ")
        maj_singular = (pynutil.delete('currency_maj: "') + self.maj_unit_singular + pynutil.delete('"'))
        maj_plural = (pynutil.delete('currency_maj: "') + self.maj_unit_plural + pynutil.delete('"'))
        integer_one = (pynutil.delete("integer_part:") + self.DELETE_SPACE + pynutil.delete('"') + self.integer_one_graph +
                       pynutil.delete('"'))
        integer_other = (pynutil.delete("integer_part:") + self.DELETE_SPACE + pynutil.delete('"') + self.integer_other_graph +
                         pynutil.delete('"'))
        zero_fraction = (self.DELETE_SPACE + pynutil.delete('fractional_part: "') + self.zero_fraction_graph +
                         pynutil.delete('"'))

        graph_integer = integer_one + zero_fraction.ques + keep_space + maj_singular
        graph_integer |= integer_other + zero_fraction.ques + keep_space + maj_plural

        trimmed_fraction = (self.DIGIT.star + (self.DIGIT - "0") + pynutil.delete("0").star) @ decimal.graph
        fractional = (pynutil.insert("point ") + pynutil.delete("fractional_part:") + self.DELETE_SPACE + pynutil.delete('"') +
                      trimmed_fraction + pynutil.delete('"'))
        optional_integer = (decimal.integer + self.DELETE_SPACE + self.INSERT_SPACE).ques
        money_decimal = (decimal.integer
                         | decimal.integer + decimal.quantity
                         | optional_integer + fractional + decimal.optional_quantity)
        graph_decimal = money_decimal + keep_space + maj_plural

        graph = graph_integer | graph_decimal

        delete_tokens = self.delete_tokens(graph)
        self.verbalizer = delete_tokens.optimize()
