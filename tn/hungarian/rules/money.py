# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import re

import pynini
from pynini.lib.pynutil import delete, insert

from tn.hungarian.rules.utils import (DIGIT, HU_ALPHA, NONZERO, SIGMA, data_labels, inflect_abbreviation,
                                      naive_inflector)
from tn.processor import Processor

TOKEN_ORDER = {"money": ["integer_part", "fractional_part", "quantity", "currency", "minor"]}


class Money(Processor):
    """Money, e.g. ``$1,01`` -> ``egy dollár egy cent``, ``1 mrd GBP-t`` -> ``egy milliárd angol fontot``.

    The currency stays raw (including Hungarian case endings such as
    ``GBP-t``). A one- or two-digit amount after a leading symbol is tagged
    as ``minor`` (read in cents); other fractions are decimal readings.
    """

    def __init__(self, cardinal, decimal):
        super().__init__("money", token_orders=TOKEN_ORDER)
        majors = [(row[0], row[1]) for row in data_labels("money/currency.tsv")]
        minors = dict((row[0], row[1]) for row in data_labels("money/currency_minor.tsv"))
        symbols = pynini.union(*[pynini.escape(symbol) for symbol, _ in majors]).optimize()
        inflected = []
        for symbol, word in majors:
            inflected.append((symbol, word))
            inflected += naive_inflector(symbol, word)
            if re.match("^[A-Z]{3}$", symbol):
                inflected += inflect_abbreviation(symbol, word)
        currency = pynini.string_map(inflected).optimize()

        integer_t = self.tag_field("integer_part", cardinal.graph)
        decimal_t = decimal.unsigned_tagger
        # After a leading symbol only long fractions or quantities read as decimals.
        long_fraction = pynini.union(DIGIT, ".", " ").plus + "," + pynini.closure(DIGIT, 2) + NONZERO + pynini.closure("0")
        prefix_decimal = (long_fraction | SIGMA + HU_ALPHA) @ decimal_t
        zero_cents_t = self.tag_field("fractional_part", pynini.closure("0", 1))
        minor_t = self.tag_field("minor", DIGIT.ques + NONZERO + pynini.closure("0"))
        prefix = self.tag_field("currency", symbols) + insert(" ") + (
            integer_t + (delete(",") + insert(" ") + (zero_cents_t | minor_t)).ques
            | prefix_decimal
        )
        space = delete(" ").ques + insert(" ")
        suffix = (integer_t | decimal_t) + space + self.tag_field("currency", currency)
        self.tagger = self.add_tokens(prefix | suffix)

        cents = (DIGIT.ques + NONZERO + delete("0").star) @ (
            delete("0") + NONZERO | NONZERO + insert("0") | NONZERO + DIGIT
        ) @ cardinal.two_digit_non_zero
        integer_v = self.verbalize_field("integer_part", cardinal.graph)
        nonzero_integer = pynini.difference(pynini.project(cardinal.graph, "input"), "0") @ cardinal.graph
        nonzero_integer_v = self.verbalize_field("integer_part", nonzero_integer)
        zero_cents_v = self.verbalize_field("fractional_part", delete(pynini.closure("0", 1)), leading_space=True)
        currency_v = pynini.accep(" ") + self.verbalize_field("currency", currency)
        amount_v = (integer_v + zero_cents_v.ques | decimal.unsigned_verbalizer) + currency_v
        minor_v = []
        for symbol, word in majors:
            minor = self.verbalize_field("minor", cents, leading_space=True) + insert(" " + minors[symbol])
            symbol = pynini.escape(symbol)
            currency_symbol_v = self.verbalize_field("currency", pynini.cross(symbol, word))
            minor_v.append(nonzero_integer_v + pynini.accep(" ") + currency_symbol_v + insert(" ") + minor)
            minor_v.append(self.verbalize_field("integer_part", delete("0")) + delete(" ")
                           + self.verbalize_field("currency", delete(symbol)) + minor)
        self.verbalizer = self.delete_tokens(amount_v | pynini.union(*minor_v))
