# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pynini.lib.pynutil import add_weight, delete, insert

from tn.italian.rules.utils import load_pairs
from tn.processor import Processor

# The money verbalizer reads the amount before naming the currency.
TOKEN_ORDER = {"money": ["integer_part", "fractional_part", "quantity", "currency"]}


class Money(Processor):
    """Amounts with a currency, e.g. ``2,01 ₽`` -> ``due rubli un copeca``.

    The tagger keeps the written amount fields and the currency symbol in
    either order (``€1`` or ``1 €``); the verbalizer names the major unit
    after the integer and reads one- or two-digit fractions as minor units.
    Longer fractions are read as decimals followed by the plural major unit.
    """

    def __init__(self, cardinal, decimal):
        super().__init__("money", token_orders=TOKEN_ORDER)
        majors = load_pairs("italian/data/money/currency_major.tsv")
        minors = dict(load_pairs("italian/data/money/currency_minor.tsv"))
        feminine = dict(load_pairs("italian/data/money/currency_plural_fem.tsv"))
        plurals = dict(load_pairs("italian/data/money/currency_plural_masc.tsv"))
        plurals.update(feminine)

        digit = self.DIGIT
        currency = pynini.union(*[pynini.escape(symbol) for symbol, _ in majors]).optimize()
        integer = cardinal.graph
        cents = digit | digit + digit
        quantity = decimal.quantity

        # Tagger: amount and currency in either order.
        integer_tag = self.tag_field("integer_part", integer)
        fraction_tag = delete(",") + insert(" ") + self.tag_field("fractional_part", digit.plus)
        quantity_tag = delete(" ") + insert(" ") + self.tag_field("quantity", quantity)
        amount_tag = integer_tag + fraction_tag.ques + quantity_tag.ques
        currency_tag = self.tag_field("currency", currency)
        space = delete(" ").ques + insert(" ")
        self.tagger = self.add_tokens(amount_tag + space + currency_tag | currency_tag + space + amount_tag)

        # Verbalizer: one branch per currency, since the currency names both
        # the major unit (after the integer) and the minor unit.
        many = (self.VSIGMA - "1") @ integer
        two = (digit + digit | digit + insert("0")) @ (pynini.closure(digit, 2, 2) - "00" - "01")
        cent_value = two @ (cardinal.two_digits | delete("0") + cardinal.digits)
        integer_verb = self.verbalize_field("integer_part", integer)
        decimal_verb = integer_verb + insert(" virgola ") + delete(" ")
        decimal_verb += self.verbalize_field("fractional_part", decimal.fractional)
        quantity_verb = delete(" ") + insert(" ") + self.verbalize_field("quantity", quantity) + insert(" di")
        branches = []
        for symbol, name in majors:
            plural = plurals.get(name, name)
            minor = minors[symbol]
            one_major = pynini.cross("1", "una" if name in feminine else "un") + insert(" " + name)
            major_graph = one_major | many + insert(" " + plural)
            major = self.verbalize_field("integer_part", major_graph)
            # ``0,50 €`` reads only the minor units.
            nonzero_major = self.verbalize_field("integer_part", (self.VSIGMA - "0") @ major_graph)
            minor_one = pynini.cross("01", "un") + insert(" " + minor)
            # A one-digit fraction counts tens of minor units (``4,2`` -> twenty pence).
            minor_many = cent_value + insert(" " + plurals.get(minor, minor))
            cents_verb = delete(" ") + self.verbalize_field("fractional_part", cents @ (minor_one | minor_many))
            no_cents = delete(" ") + self.verbalize_field("fractional_part", pynini.cross("00", ""))
            zero_major = self.verbalize_field("integer_part", delete("0"))
            amount = pynini.union(
                major,
                nonzero_major + insert(" ") + cents_verb,
                major + no_cents,
                zero_major + cents_verb,
                add_weight(decimal_verb + insert(" " + plural), 0.01),
                (integer_verb | decimal_verb) + quantity_verb + insert(" " + plural),
            )
            symbol_verb = delete(" ") + self.verbalize_field("currency", pynini.cross(pynini.escape(symbol), ""))
            branches.append(amount + symbol_verb)
        self.verbalizer = self.delete_tokens(pynini.union(*branches).optimize())
