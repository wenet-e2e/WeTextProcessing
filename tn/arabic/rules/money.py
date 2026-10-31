# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from tn.processor import Processor
from tn.utils import get_abs_path, load_labels

# Amounts are read before the currency, whichever side the symbol was written on.
TOKEN_ORDER = {"money": ["integer_part", "currency_maj", "fractional_part"]}


def first_labels(path):
    """Maps each written symbol to its first listed reading."""

    labels = {}
    for row in load_labels(get_abs_path(path)):
        labels.setdefault(row[0], row[1])
    return labels


def written(symbol):
    """Accepts a currency symbol literally (no bracket parsing)."""

    return pynini.accep(pynini.escape(symbol))


class Money(Processor):
    """Money, e.g. ``$7.05`` -> ``سبعة دولار وخمسة سنتات`` and ``5د.ك`` -> ``خمسة دينار كويتي``.

    The tagger keeps the written currency symbol and amount digits. The
    verbalizer reads the amount, the currency name and, when the currency has
    one, the minor unit (plural for 2..10, singular otherwise).
    """

    def __init__(self, cardinal):
        super().__init__("money", token_orders=TOKEN_ORDER)
        self.cardinal = cardinal
        self.build_graphs()
        self.build_tagger()
        self.build_verbalizer()

    def build_graphs(self):
        number = self.cardinal.graph_with_leading_zeros
        nonzero = self.DIGIT - "0"
        self.majors = first_labels("arabic/data/money/currency_major.tsv")
        self.locals = first_labels("arabic/data/money/local_currency.tsv")
        self.minor_singular = first_labels("arabic/data/money/currency_minor_singular.tsv")
        self.minor_plural = first_labels("arabic/data/money/currency_minor_plural.tsv")

        integer = self.DIGIT.plus - "0"
        self.integer = (integer @ number).optimize()
        self.integer_not_one = ((integer - "1") @ number).optimize()
        # Minor digits keep at most two significant places: 5 -> 50, 05 -> 5, 500 -> 50.
        minor_digits = self.DIGIT.star + nonzero + delete("0").star
        minor_digits @= delete("0") + nonzero | nonzero + insert("0") | nonzero + self.DIGIT
        self.minor_digits = minor_digits.optimize()
        two_to_ten = pynini.union(*[str(n) for n in range(2, 11)])
        self.minor_plural_amount = (self.minor_digits @ two_to_ten @ number).optimize()
        self.minor_singular_amount = (self.minor_digits @ (self.DIGIT.plus - two_to_ten) @ number).optimize()
        self.minor_amount = (self.minor_digits @ number).optimize()
        self.zeros = pynini.accep("0").plus.optimize()

    def minor(self, symbol):
        """Fractional readings for one major currency; None if it has no minor unit."""

        readings = []
        if symbol in self.minor_singular:
            readings.append(self.minor_singular_amount + insert(" " + self.minor_singular[symbol]))
        if symbol in self.minor_plural:
            readings.append(self.minor_plural_amount + insert(" " + self.minor_plural[symbol]))
        return pynini.union(*readings).optimize() if readings else None

    def build_tagger(self):
        majors = pynini.union(*[written(symbol) for symbol in self.majors])
        locals_ = pynini.union(*[written(symbol) for symbol in self.locals])
        currency = self.tag_field("currency_maj", majors) + insert(" ")
        fractional = delete(".") + insert(" ") + self.tag_field("fractional_part", self.minor_digits)
        prefix = self.tag_field("integer_part", self.integer_not_one)
        zeros = delete(".") + insert(" ") + self.tag_field("fractional_part", self.zeros)
        prefix |= self.tag_field("integer_part", pynini.accep("1")) + zeros.ques
        prefix |= self.tag_field("integer_part", self.integer) + fractional
        cents = []
        for symbol in self.majors:
            minor = self.minor(symbol)
            if minor is not None:
                cents.append(self.tag_field("currency_maj", written(symbol)) + insert(" ")
                             + self.tag_field("integer_part", pynini.accep("0")) + delete(".") + insert(" ")
                             + self.tag_field("fractional_part", minor))
        suffix = self.tag_field("integer_part", self.integer_not_one) + insert(" ")
        suffix += self.tag_field("currency_maj", locals_)
        self.tagger = self.add_tokens(currency + prefix | pynini.union(*cents) | suffix)

    def build_verbalizer(self):
        and_ = insert(" و")
        readings = []
        for symbol, name in self.majors.items():
            currency = self.verbalize_field("currency_maj", pynini.cross(written(symbol), name), True)
            delete_currency = self.verbalize_field("currency_maj", delete(written(symbol)), True)
            integer = self.verbalize_field("integer_part", self.integer)
            plain = self.verbalize_field("integer_part", self.integer_not_one) + insert(" ") + currency
            # "$1" and "$1.00" put the currency first: دولار واحد.
            one = self.verbalize_field("integer_part", delete("1")) + currency + insert(" واحد")
            one += self.verbalize_field("fractional_part", delete(self.zeros), True).ques
            fractional = add_weight(self.minor_amount, 0.0001)
            minor = self.minor(symbol)
            if minor is not None:
                fractional |= minor
                cents = self.verbalize_field("integer_part", delete("0")) + delete_currency
                cents += self.verbalize_field("fractional_part", minor, True)
                readings.append(cents)
            with_minor = integer + insert(" ") + currency + and_
            with_minor += self.verbalize_field("fractional_part", fractional, True)
            readings.extend([plain, one, with_minor])
        for symbol, name in self.locals.items():
            currency = self.verbalize_field("currency_maj", pynini.cross(written(symbol), name), True)
            readings.append(self.verbalize_field("integer_part", self.integer_not_one) + insert(" ") + currency)
        self.verbalizer = self.delete_tokens(pynini.union(*readings))
