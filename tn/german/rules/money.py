# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pynini
from pynini.lib import pynutil

from tn.processor import Processor
from tn.utils import get_abs_path, load_labels


TOKEN_ORDER = {
    "money": ["kind", "negative", "integer_part", "fractional_part", "quantity", "currency"],
}


class Money(Processor):
    """Raw currency and flat amounts; currency-dependent readings belong to V."""

    def __init__(self, cardinal, decimal, deterministic=True):
        super().__init__("money", token_orders=TOKEN_ORDER)
        self.deterministic = deterministic
        major_labels = load_labels(get_abs_path("german/data/money/currency.tsv"))
        minor_singular = dict(load_labels(get_abs_path("german/data/money/currency_minor_singular.tsv")))
        minor_plural = dict(load_labels(get_abs_path("german/data/money/currency_minor_plural.tsv")))
        currencies = {}
        for symbol, word in major_labels:
            currencies.setdefault(symbol, []).append(word.replace("\u00a0", " "))

        change_one = pynini.cdrewrite(pynini.cross("eins", "ein"), "[BOS]", "[EOS]", self.VSIGMA)
        major_number = (cardinal.graph @ change_one).optimize()
        nonzero_number = ((pynini.closure(self.DIGIT, 1) - "0") @ major_number).optimize()
        zeros = pynini.closure(pynutil.delete("0"), 1)
        major_amount_t = self.tag_field("integer_part", major_number) + pynini.closure(
            pynutil.delete(",") + pynutil.insert(" ") + self.tag_field("fractional_part", zeros), 0, 1,
        )
        major_amount_v = self.verbalize_field("integer_part", major_number) + pynini.closure(
            self.verbalize_field("fractional_part", zeros, leading_space=True), 0, 1,
        )

        # Keep every written fractional digit in T. Only V trims and pads cents.
        cents = (
            pynini.closure(self.DIGIT) + (self.DIGIT - "0") + pynini.closure(pynutil.delete("0"))
        ) @ (
            pynutil.delete("0") + (self.DIGIT - "0")
            | (self.DIGIT - "0") + pynutil.insert("0")
            | (self.DIGIT - "0") + self.DIGIT
        )
        cents_number = (cents @ cardinal.two_digits).optimize()
        cents_one = (cents @ pynini.cross("1", "ein")).optimize()
        cents_other = (cents @ (pynini.closure(self.DIGIT, 1, 2) - "1") @ cardinal.two_digits).optimize()
        cents_with_name = (cents_one | cents_other).optimize()
        minor_amount_t = (
            self.tag_field("integer_part", nonzero_number | pynutil.delete("0"))
            + pynutil.delete(",") + pynutil.insert(" ") + self.tag_field("fractional_part", cents_with_name)
        )
        bare_amount_t = (
            self.tag_field("integer_part", nonzero_number) + pynutil.delete(",")
            + pynutil.insert(" ") + self.tag_field("fractional_part", cents_number)
        )

        # The source decimal path accepts >2 significant fractional places, or
        # any supported quantity. Its trailing zeros are removed only without q.
        long_decimal = (
            pynini.closure(self.DIGIT, 1) + "," + pynini.closure(self.DIGIT, 2)
            + (self.DIGIT - "0") + pynini.closure("0")
        )
        with_quantity = self.VSIGMA + self.ALPHA
        decimal_amount_t = (long_decimal | with_quantity) @ decimal.fields_tagger
        trim_zeros = (
            pynini.closure(self.DIGIT, 2) + (self.DIGIT - "0") + pynini.closure(pynutil.delete("0"))
        )
        decimal_no_quantity_v = (
            self.verbalize_field("integer_part", decimal.integer_graph) + pynutil.insert(" komma ")
            + self.verbalize_field("fractional_part", trim_zeros @ decimal.fractional_graph, leading_space=True)
        )
        quantity_v = (
            self.verbalize_field("quantity", decimal.quantity_graph, leading_space=True)
        )
        sign_v = pynini.closure(self.verbalize_field("negative", cardinal.sign) + pynutil.delete(" "), 0, 1)
        decimal_quantity_v = sign_v + self.verbalize_field("integer_part", decimal.integer_graph) + (
            pynutil.insert(" ") + quantity_v
            | pynutil.insert(" komma ")
            + self.verbalize_field("fractional_part", decimal.fractional_graph, leading_space=True)
            + pynutil.insert(" ") + quantity_v
        )
        decimal_amount_v = (decimal_no_quantity_v | decimal_quantity_v).optimize()

        taggers = []
        verbalizers = []
        for symbol, words in currencies.items():
            currency_t = self.tag_field("currency", pynini.accep(symbol)) + pynutil.insert(" ")
            currency_v = self.verbalize_field("currency", pynutil.delete(symbol), leading_space=True)
            # Raw fields collapse equal-cost source tags. Stabilize their order
            # by TSV row, without dropping any alternate currency readings.
            major_word = pynini.union(*[
                pynutil.add_weight(pynini.accep(word), index * 0.00001)
                for index, word in enumerate(words)
            ])

            def kind_t(kind):
                return pynutil.insert('kind: "' + kind + '" ')

            def kind_v(kind):
                return pynutil.delete('kind: "' + kind + '" ')

            taggers.extend([
                currency_t + kind_t("major") + major_amount_t,
                currency_t + kind_t("decimal") + decimal_amount_t,
                pynutil.add_weight(currency_t + kind_t("minor_bare") + bare_amount_t, 0.0001),
            ])
            verbalizers.extend([
                kind_v("major") + major_amount_v + pynutil.insert(" ") + pynutil.insert(major_word) + currency_v,
                kind_v("decimal") + decimal_amount_v + pynutil.insert(" ") + pynutil.insert(major_word) + currency_v,
                kind_v("minor_bare") + self.verbalize_field("integer_part", nonzero_number)
                + pynutil.insert(" ") + pynutil.insert(major_word) + pynutil.insert(" ")
                + self.verbalize_field("fractional_part", cents_number, leading_space=True) + currency_v,
            ])
            if symbol in minor_singular and symbol in minor_plural:
                taggers.append(currency_t + kind_t("minor") + minor_amount_t)
                minor_v = (
                    self.verbalize_field("fractional_part", cents_one, leading_space=True)
                    + pynutil.insert(" " + minor_singular[symbol])
                    | self.verbalize_field("fractional_part", cents_other, leading_space=True)
                    + pynutil.insert(" " + minor_plural[symbol])
                )
                verbalizers.append(
                    kind_v("minor") + (
                        self.verbalize_field("integer_part", nonzero_number) + pynutil.insert(" ")
                        + pynutil.insert(major_word) + pynutil.insert(" ")
                        + pynini.closure(pynutil.insert("und "), 0, 1) + minor_v
                        | self.verbalize_field("integer_part", pynutil.delete("0")) + minor_v
                    ) + currency_v
                )

        self.tagger = self.add_tokens(pynini.union(*taggers))
        self.verbalizer = self.delete_tokens(pynini.union(*verbalizers))
