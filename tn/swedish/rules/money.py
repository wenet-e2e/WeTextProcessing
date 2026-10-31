# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pynini.lib import pynutil

from tn.processor import Processor
from tn.swedish.rules.utils import alt, labels, tsv

TOKEN_ORDER = {
    "money": ["integer_part", "fractional_part", "quantity", "currency_maj", "minor_part"],
}


class Money(Processor):
    """Raw amounts and currency symbols; gender and number agreement is verbalized."""

    def __init__(self, cardinal, decimal, deterministic=True):
        super().__init__("money", token_orders=TOKEN_ORDER)
        self.deterministic = deterministic
        digit = self.DIGIT
        nonzero = digit - "0"
        plurals = tsv("money/currency_plurals")
        minor_singular = dict((row[0], row[1]) for row in labels("money/currency_minor_singular"))
        minor_plural = dict((row[0], row[1]) for row in labels("money/currency_minor_plural"))
        genders = {
            "en": (labels("money/currency_major"), pynini.cross("1", "en"), cardinal.graph_no_one_en),
            "ett": (labels("money/currency_major_nt"), pynini.cross("1", "ett"), cardinal.graph_no_one),
        }
        all_symbols = pynini.union(*[row[0] for rows, _, _ in genders.values() for row in rows])
        minor_symbols = pynini.union(*[
            row[0] for rows, _, _ in genders.values() for row in rows
            if row[0] in minor_singular or row[0] in minor_plural])

        # Two significant fractional digits, e.g. 5 -> 50, 05 -> 5, 500 -> 50.
        two_digits = (digit.star + nonzero + pynutil.delete("0").star) @ (
            pynutil.delete("0") + nonzero | nonzero + pynutil.insert("0") | nonzero + digit)
        minor_one = (two_digits @ pynini.cross("1", "en")).optimize()
        minor_other = (two_digits @ (pynini.closure(digit, 1, 2) - "1") @ cardinal.up_to_thousand_en).optimize()
        zeros = pynutil.delete(pynini.accep("0").plus)

        # Decimal amounts: three significant fractional digits, or a quantity word.
        long_fraction = (pynini.closure(digit, 2) + nonzero + pynutil.delete("0").star) @ decimal.fractional
        long_fraction_t = pynutil.delete(",") + pynutil.insert(" ") + self.tag_field("fractional_part", long_fraction)
        long_fraction_v = pynutil.insert(" komma ") + self.verbalize_field(
            "fractional_part", long_fraction, leading_space=True)

        def integer_t(graph):
            return self.tag_field("integer_part", graph)

        def integer_v(graph):
            return self.verbalize_field("integer_part", graph)

        def currency_t(symbols):
            return self.tag_field("currency_maj", symbols)

        def currency_v(graph, leading_space=True):
            return self.verbalize_field("currency_maj", graph, leading_space=leading_space)

        space = pynini.accep(" ") | pynutil.insert(" ")
        zeros_t = pynutil.delete(",") + pynutil.insert(" ") + self.tag_field("minor_part", pynini.accep("0").plus)
        zeros_v = self.verbalize_field("minor_part", zeros, leading_space=True)
        quantity_t, _ = decimal.quantity_fields()
        amount_t = integer_t(cardinal.graph) + zeros_t.ques
        decimal_t = quantity_t | integer_t(cardinal.graph) + long_fraction_t
        minor_t = pynutil.delete(",") + pynutil.insert(" ") + self.tag_field("minor_part", minor_one | minor_other)
        nonzero_amount = cardinal.not_zero @ cardinal.graph
        with_minor_t = (
            integer_t(nonzero_amount) + minor_t
            | (integer_t(pynini.accep("0")) + pynutil.insert(" ")).ques + pynutil.delete(",") + self.tag_field(
                "minor_part", minor_one | minor_other)
        )
        tagger = (
            currency_t(all_symbols) + space + (amount_t | decimal_t)
            | (amount_t | decimal_t) + space + currency_t(all_symbols)
            | pynutil.add_weight(currency_t(minor_symbols) + pynutil.insert(" ") + with_minor_t, -0.0001)
        )

        paths = []
        for gender, (rows, one, other) in genders.items():
            singular = pynini.string_map(rows)
            plural = singular @ plurals
            en = gender == "en"
            decimal_fields = decimal.quantity_fields()[1] | integer_v(
                cardinal.graph_en if en else cardinal.graph) + long_fraction_v
            paths.append(integer_v(one) + pynutil.insert(" ") + currency_v(singular) + zeros_v.ques)
            paths.append(integer_v(other) + pynutil.insert(" ") + currency_v(plural) + zeros_v.ques)
            paths.append(decimal_fields + pynutil.insert(" ") + currency_v(plural))
            singular_minor, plural_minor = [], []
            for row in rows:
                symbol = row[0]
                words = []
                if symbol in minor_singular:
                    words.append(minor_one + pynutil.insert(" " + minor_singular[symbol]))
                if symbol in minor_plural:
                    words.append(minor_other + pynutil.insert(" " + minor_plural[symbol]))
                if not words:
                    continue
                words = pynini.union(*words)
                minor = self.verbalize_field("minor_part", words, leading_space=True)
                if not deterministic:
                    joined = alt(pynutil.insert("och ") + words) | alt(pynutil.insert("komma ") + words)
                    minor |= self.verbalize_field("minor_part", joined, leading_space=True)
                    minor |= alt(self.verbalize_field("minor_part", minor_one | minor_other, leading_space=True))
                minor = pynutil.insert(" ") + minor
                singular_minor.append(currency_v(pynini.accep(symbol) @ singular) + minor)
                plural_minor.append(currency_v(pynini.accep(symbol) @ plural) + minor)
                drop = pynutil.delete(symbol)
                minor_only = integer_v(pynutil.delete("0")) + currency_v(drop) | currency_v(drop, leading_space=False)
                paths.append(minor_only + self.verbalize_field("minor_part", words, leading_space=True))
            nonzero_other = cardinal.not_zero @ other
            paths.append(integer_v(one) + pynutil.insert(" ") + pynini.union(*singular_minor))
            paths.append(integer_v(nonzero_other) + pynutil.insert(" ") + pynini.union(*plural_minor))
        self.tagger = self.add_tokens(tagger)
        self.verbalizer = self.delete_tokens(pynini.union(*paths))
