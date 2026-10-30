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
from tn.spanish.rules.utils import (
    DECIMAL_SEPARATOR,
    DIGIT,
    data_labels,
    restrict,
    shift_cardinal_gender,
    shift_number_gender,
    strip_cardinal_apocope,
)

# The currency is read between the integer and the fractional part
# ("cinco dólares cincuenta centavos"), so it sits between them on the wire.
TOKEN_ORDER = {"money": ["integer_part", "currency", "fractional_part", "quantity"]}


def _currencies():
    """Currency symbols with the names and genders of their major and minor units."""
    plurals, genders = {}, {}
    for gender in ("fem", "masc"):
        for singular, plural in data_labels("money/currency_plural_" + gender + ".tsv"):
            plurals[singular] = plural
            genders[singular] = gender
    minors = dict(data_labels("money/currency_minor.tsv"))
    currencies = {}
    for symbol, major in data_labels("money/currency_major.tsv"):
        if major not in plurals:
            continue
        minor = minors.get(symbol)
        currencies[symbol] = {
            "gender": genders[major],
            "major": (major, plurals[major]),
            "minor": (minor, plurals[minor], genders[minor]) if minor in plurals else None,
        }
    return currencies


class Money(Processor):
    """Raw amount digits and the written currency symbol; V names the currency."""

    def __init__(self, cardinal, decimal, deterministic=True):
        super().__init__("money", token_orders=TOKEN_ORDER)
        self.deterministic = deterministic
        currencies = _currencies()
        insert_space = pynutil.insert(" ")
        optional_space = pynini.closure(pynutil.delete(" "), 0, 1)

        # Tagger: the currency symbol before or after the written amount.
        currency_t = self.tag_field("currency", pynini.union(*currencies))
        integer_t = self.tag_field("integer_part", cardinal.graph)
        fractional_t = self.tag_field("fractional_part", pynini.closure(DIGIT, 1))
        number_t = integer_t + pynini.closure(pynutil.delete(DECIMAL_SEPARATOR) + insert_space + fractional_t, 0, 1)
        quantity_t = (
            self.tag_field("integer_part", decimal.quantity_integer_graph)
            | integer_t + pynutil.delete(DECIMAL_SEPARATOR) + insert_space + fractional_t
        ) + pynini.accep(" ") + self.tag_field("quantity", decimal.quantity_graph)
        self.tagger = self.add_tokens(
            currency_t + optional_space + insert_space + (number_t | quantity_t)
            | number_t + optional_space + insert_space + currency_t
        )

        # Verbalizer. Amounts agree in gender and number with the currency.
        one = pynini.accep("1")
        zero = pynini.accep("0")
        integers = {"masc": cardinal.masc, "fem": cardinal.fem}
        integer_v = {}
        for gender, graph in integers.items():
            integer_v[gender, "one"] = self.verbalize_field("integer_part", one @ graph)
            integer_v[gender, "zero"] = self.verbalize_field("integer_part", zero @ graph)
            integer_v[gender, "other"] = self.verbalize_field("integer_part", restrict(graph, one | zero))
        decimal_integers = {"masc": decimal.integer_graph, "fem": shift_cardinal_gender(decimal.integer_graph)}
        decimal_fractions = {"masc": decimal.fractional_graph, "fem": shift_number_gender(decimal.fractional_graph)}
        decimal_v = {}
        for gender, graph in decimal_integers.items():
            decimal_v[gender, "one"] = self.verbalize_field("integer_part", one @ graph)
            decimal_v[gender, "other"] = self.verbalize_field("integer_part", restrict(graph, one))
        quantity_integer_v = self.verbalize_field("integer_part", decimal.quantity_integer_graph)
        quantity_v = self.verbalize_field("quantity", decimal.quantity_graph, leading_space=True)

        # Cents: trailing zeros dropped, then two digits ("5" and "50" are 50).
        non_zero = pynini.difference(DIGIT, "0")
        two_digits = (pynini.closure(DIGIT) + non_zero + pynini.closure(pynutil.delete("0"))) @ pynini.union(
            pynutil.delete("0") + non_zero, non_zero + pynutil.insert("0"), non_zero + DIGIT,
        )
        cents = (two_digits @ cardinal.two_digit_non_zero).optimize()
        cents_one = (two_digits @ one @ cardinal.two_digit_non_zero).optimize()
        cents_other = (two_digits @ pynini.difference(pynini.closure(DIGIT, 1, 2), one) @ cardinal.two_digit_non_zero).optimize()
        bare_cents = {"masc": strip_cardinal_apocope(cents), "fem": shift_cardinal_gender(cents)}
        cents_by_gender = {
            "masc": (cents_one, cents_other),
            "fem": (shift_cardinal_gender(cents_one), shift_cardinal_gender(cents_other)),
        }

        conjunction = pynini.accep("")
        if not deterministic:
            conjunction |= pynutil.add_weight(pynutil.insert(pynini.union("con ", "y ")), 0.001)

        def currency_v(symbol, name):
            return self.verbalize_field("currency", pynini.cross(symbol, name), leading_space=True)

        def minor_v(symbol):
            minor = currencies[symbol]["minor"]
            if minor is None:
                return None
            singular, plural, gender = minor
            cents_one, cents_other = cents_by_gender[gender]
            return self.verbalize_field(
                "fractional_part",
                cents_one + pynutil.insert(" " + singular) | cents_other + pynutil.insert(" " + plural),
                leading_space=True,
            )

        branches = []
        for gender in ("masc", "fem"):
            symbols = [symbol for symbol, currency in currencies.items() if currency["gender"] == gender]
            for number in ("one", "zero", "other"):
                tails = []
                for symbol in symbols:
                    singular, plural = currencies[symbol]["major"]
                    name = singular if number == "one" else plural
                    tail = insert_space + currency_v(symbol, name)
                    if number != "zero":
                        # "cinco dólares cincuenta centavos" or "cinco dólares cincuenta".
                        cents_v = pynutil.add_weight(
                            self.verbalize_field("fractional_part", bare_cents[gender], leading_space=True), 0.0001,
                        )
                        minor = minor_v(symbol)
                        if minor is not None:
                            cents_v |= minor
                        tail += pynini.closure(insert_space + conjunction + cents_v, 0, 1)
                    tails.append(tail)
                branches.append(integer_v[gender, number] + pynini.union(*tails))

            # "uno coma cero cero uno dólares".
            for number in ("one", "other"):
                tails = []
                for symbol in symbols:
                    singular, plural = currencies[symbol]["major"]
                    name = singular if number == "one" else plural
                    tails.append(
                        self.verbalize_field("currency", pynutil.delete(symbol), leading_space=True)
                        + pynutil.insert(" coma ")
                        + self.verbalize_field("fractional_part", decimal_fractions[gender], leading_space=True)
                        + pynutil.insert(" " + name)
                    )
                branches.append(pynutil.add_weight(decimal_v[gender, number] + pynini.union(*tails), 0.001))

        # Only cents: "cincuenta y cinco centavos".
        minors = []
        for symbol in currencies:
            minor = minor_v(symbol)
            if minor is not None:
                minors.append(self.verbalize_field("currency", pynutil.delete(symbol), leading_space=True) + minor)
        branches.append(self.verbalize_field("integer_part", pynutil.delete("0")) + pynini.union(*minors))

        # Quantities are masculine nouns: "dos coma tres billones de euros".
        integer_tails, decimal_tails = [], []
        for symbol, currency in currencies.items():
            name = pynutil.insert(" ") + quantity_v + pynutil.insert(" de " + currency["major"][1])
            delete_currency = self.verbalize_field("currency", pynutil.delete(symbol), leading_space=True)
            integer_tails.append(delete_currency + name)
            decimal_tails.append(
                delete_currency + pynutil.insert(" coma ")
                + self.verbalize_field("fractional_part", decimal_fractions["masc"], leading_space=True) + name
            )
        branches.append(quantity_integer_v + pynini.union(*integer_tails))
        branches.append((decimal_v["masc", "one"] | decimal_v["masc", "other"]) + pynini.union(*decimal_tails))
        self.verbalizer = self.delete_tokens(pynini.union(*branches))
