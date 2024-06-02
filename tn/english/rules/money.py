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

from tn.processor import Processor
from tn.utils import get_abs_path, load_labels
from tn.english.rules.cardinal import Cardinal
from tn.english.rules.decimal import Decimal
from tn.english.rules.measure import SINGULAR_TO_PLURAL

min_singular = pynini.string_file(
    get_abs_path("english/data/money/currency_minor_singular.tsv"))
min_plural = pynini.string_file(
    get_abs_path("english/data/money/currency_minor_plural.tsv"))
maj_singular = pynini.string_file(
    (get_abs_path("english/data/money/currency_major.tsv")))


class Money(Processor):

    def __init__(self, deterministic: bool = False):
        """
        Args:
            deterministic: if True will provide a single transduction option,
                for False multiple transduction are generated (used for audio-based normalization)
        """
        super().__init__('money', ordertype="en_tn")
        self.deterministic = deterministic
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        """
        Finite state transducer for classifying money, suppletive aware, e.g.
            $12.05 -> money { integer_part: "twelve" currency_maj: "dollars" fractional_part: "five" currency_min: "cents" preserve_order: \"true\" }
            $12.0500 -> money { integer_part: "twelve" currency_maj: "dollars" fractional_part: "five" currency_min: "cents" preserve_order: \"true\" }
            $1 -> money { currency_maj: "dollar" integer_part: "one" }
            $1.00 -> money { currency_maj: "dollar" integer_part: "one" }
            $0.05 -> money { fractional_part: "five"  currency_min: "cents" preserve_order: \"true\" }
            $1 million -> money { currency_maj: "dollars" integer_part: "one" quantity: "million" }
            $1.2 million -> money { currency_maj: "dollars" integer_part: "one" fractional_part: "two" quantity: "million" }
            $1.2320 -> money { currency_maj: "dollars" integer_part: "one" fractional_part: "two three two" }
        """
        cardinal = Cardinal(self.deterministic)
        decimal = Decimal(self.deterministic)
        cardinal_graph = cardinal.graph_with_and
        graph_decimal_final = decimal.final_graph_wo_negative_w_abbr

        maj_singular_labels = load_labels(
            get_abs_path("english/data/money/currency_major.tsv"))
        maj_unit_plural = maj_singular @ SINGULAR_TO_PLURAL
        maj_unit_singular = maj_singular

        graph_maj_singular = pynutil.insert(
            "currency_maj: \"") + maj_unit_singular + pynutil.insert("\"")
        graph_maj_plural = pynutil.insert(
            "currency_maj: \"") + maj_unit_plural + pynutil.insert("\"")

        optional_delete_fractional_zeros = pynini.closure(
            pynutil.delete(".") + pynini.closure(pynutil.delete("0"), 1), 0, 1)

        graph_integer_one = pynutil.insert("integer_part: \"") + pynini.cross(
            "1", "one") + pynutil.insert("\"")
        # only for decimals where third decimal after comma is non-zero or with quantity
        decimal_delete_last_zeros = (
            pynini.closure(self.DIGIT | pynutil.delete(",")) +
            pynini.accep(".") + pynini.closure(self.DIGIT, 2) +
            (self.DIGIT - "0") + pynini.closure(pynutil.delete("0")))
        decimal_with_quantity = pynini.closure(self.VCHAR) + self.ALPHA

        graph_decimal = (graph_maj_plural + self.INSERT_SPACE +
                         (decimal_delete_last_zeros | decimal_with_quantity)
                         @ graph_decimal_final)

        graph_integer = (
            pynutil.insert("integer_part: \"") +
            ((pynini.closure(self.VCHAR) - "1") @ cardinal_graph) +
            pynutil.insert("\""))  # noqa

        graph_integer_only = graph_maj_singular + self.INSERT_SPACE + graph_integer_one
        graph_integer_only |= graph_maj_plural + self.INSERT_SPACE + graph_integer

        final_graph = (graph_integer_only +
                       optional_delete_fractional_zeros) | graph_decimal

        # remove trailing zeros of non zero number in the first 2 digits and fill up to 2 digits
        # e.g. 2000 -> 20, 0200->02, 01 -> 01, 10 -> 10
        # not accepted: 002, 00, 0,
        two_digits_fractional_part = (
            pynini.closure(self.DIGIT) +
            (self.DIGIT - "0") + pynini.closure(pynutil.delete("0"))) @ (
                (pynutil.delete("0") + (self.DIGIT - "0"))
                | ((self.DIGIT - "0") + pynutil.insert("0"))
                | ((self.DIGIT - "0") + self.DIGIT))

        graph_min_singular = pynutil.insert(
            "currency_min: \"") + min_singular + pynutil.insert("\"")
        graph_min_plural = pynutil.insert(
            "currency_min: \"") + min_plural + pynutil.insert("\"")
        # format ** dollars ** cent
        decimal_graph_with_minor = None
        integer_graph_reordered = None
        decimal_default_reordered = None
        for curr_symbol, _ in maj_singular_labels:
            preserve_order = pynutil.insert(" preserve_order: \"true\"")
            integer_plus_maj = graph_integer + self.INSERT_SPACE + pynutil.insert(
                curr_symbol) @ graph_maj_plural
            integer_plus_maj |= graph_integer_one + self.INSERT_SPACE + pynutil.insert(
                curr_symbol) @ graph_maj_singular

            integer_plus_maj_with_comma = pynini.compose(
                self.DIGIT - "0" +
                pynini.closure(self.DIGIT | pynutil.delete(",")),
                integer_plus_maj)
            integer_plus_maj = pynini.compose(
                pynini.closure(self.DIGIT) - "0", integer_plus_maj)
            integer_plus_maj |= integer_plus_maj_with_comma

            graph_fractional_one = two_digits_fractional_part @ pynini.cross(
                "1", "one")
            graph_fractional_one = pynutil.insert(
                "fractional_part: \"") + graph_fractional_one + pynutil.insert(
                    "\"")
            graph_fractional = (two_digits_fractional_part @ (
                pynini.closure(self.DIGIT, 1, 2) - "1"
            ) @ cardinal.graph_hundred_component_at_least_one_none_zero_digit)
            graph_fractional = pynutil.insert(
                "fractional_part: \"") + graph_fractional + pynutil.insert(
                    "\"")

            fractional_plus_min = graph_fractional + self.INSERT_SPACE + pynutil.insert(
                curr_symbol) @ graph_min_plural
            fractional_plus_min |= (
                graph_fractional_one + self.INSERT_SPACE +
                pynutil.insert(curr_symbol) @ graph_min_singular)

            decimal_graph_with_minor_curr = integer_plus_maj + pynini.cross(
                ".", " ") + fractional_plus_min

            if not self.deterministic:
                decimal_graph_with_minor_curr |= pynutil.add_weight(
                    integer_plus_maj + pynini.cross(".", " ") +
                    pynutil.insert("fractional_part: \"") +
                    two_digits_fractional_part @ cardinal.
                    graph_hundred_component_at_least_one_none_zero_digit +
                    pynutil.insert("\""),
                    weight=0.0001,
                )
                default_fraction_graph = (
                    decimal_delete_last_zeros
                    | decimal_with_quantity) @ graph_decimal_final
            decimal_graph_with_minor_curr |= (
                pynini.closure(pynutil.delete("0"), 0, 1) +
                pynutil.delete(".") + fractional_plus_min)
            decimal_graph_with_minor_curr = (pynutil.delete(curr_symbol) +
                                             decimal_graph_with_minor_curr +
                                             preserve_order)

            decimal_graph_with_minor = (
                decimal_graph_with_minor_curr
                if decimal_graph_with_minor is None else pynini.union(
                    decimal_graph_with_minor,
                    decimal_graph_with_minor_curr).optimize())

            if not self.deterministic:
                integer_graph_reordered_curr = (pynutil.delete(curr_symbol) +
                                                integer_plus_maj +
                                                preserve_order).optimize()

                integer_graph_reordered = (
                    integer_graph_reordered_curr
                    if integer_graph_reordered is None else pynini.union(
                        integer_graph_reordered,
                        integer_graph_reordered_curr).optimize())
                decimal_default_reordered_curr = (
                    pynutil.delete(curr_symbol) + default_fraction_graph +
                    self.INSERT_SPACE +
                    pynutil.insert(curr_symbol) @ graph_maj_plural)

                decimal_default_reordered = (
                    decimal_default_reordered_curr
                    if decimal_default_reordered is None else pynini.union(
                        decimal_default_reordered,
                        decimal_default_reordered_curr)).optimize()

        # weight for SH
        final_graph |= pynutil.add_weight(decimal_graph_with_minor, -0.0001)

        if not self.deterministic:
            final_graph |= pynutil.add_weight(
                integer_graph_reordered | decimal_default_reordered, -0.1)
            # to handle "$2.00" cases
            final_graph |= pynini.compose(
                pynini.closure(self.VCHAR) + pynutil.delete(".") +
                pynini.closure(pynutil.delete("0"), 1),
                integer_graph_reordered)
        final_graph = self.add_tokens(final_graph.optimize())
        self.tagger = final_graph.optimize()

    def build_verbalizer(self):
        """
        Finite state transducer for verbalizing money, e.g.
            money { integer_part: "twelve" fractional_part: "o five" currency: "dollars" } -> twelve o five dollars
        """
        decimal = Decimal(self.deterministic)
        delete_preserve_order = pynini.closure(
            pynutil.delete(" preserve_order: \"true\"")
            | (pynutil.delete(" field_order: \"") + self.NOT_QUOTE +
               pynutil.delete("\"")))
        keep_space = pynini.accep(" ")
        maj = pynutil.delete("currency_maj: \"") + pynini.closure(
            self.NOT_QUOTE, 1) + pynutil.delete("\"")
        min = pynutil.delete("currency_min: \"") + pynini.closure(
            self.NOT_QUOTE, 1) + pynutil.delete("\"")

        fractional_part = (pynutil.delete("fractional_part: \"") +
                           pynini.closure(self.NOT_QUOTE, 1) +
                           pynutil.delete("\""))

        integer_part = decimal.integer

        #  *** currency_maj
        graph_integer = integer_part + keep_space + maj

        #  *** currency_maj + (***) | ((and) *** current_min)
        fractional = fractional_part + self.DELETE_EXTRA_SPACE + min

        if not self.deterministic:
            fractional |= pynutil.insert("and ") + fractional

        graph_integer_with_minor = integer_part + keep_space + maj + keep_space + fractional + delete_preserve_order

        # *** point *** currency_maj
        graph_decimal = decimal.numbers + keep_space + maj

        # *** current_min
        graph_minor = fractional_part + self.DELETE_EXTRA_SPACE + min + delete_preserve_order

        graph = graph_integer | graph_integer_with_minor | graph_decimal | graph_minor

        if not self.deterministic:
            graph |= graph_integer + delete_preserve_order

        delete_tokens = self.delete_tokens(graph)
        self.verbalizer = delete_tokens.optimize()
