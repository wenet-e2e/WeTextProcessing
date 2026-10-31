# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from pynini.lib.pynutil import delete, insert

from itn.utils import get_abs_path
from tn.processor import Processor
from tn.utils import load_labels

# The currency symbol is written first: $7.50, د.ك5.
TOKEN_ORDER = {"money": ["currency", "integer_part", "fractional_part", "currency_min"]}


def symbols(*paths):
    """Maps each spoken currency name to the first symbol listed for it."""

    names = {}
    for path in paths:
        for row in load_labels(get_abs_path(path)):
            names.setdefault(row[1], row[0])
    return pynini.string_map([(pynini.escape(name), pynini.escape(symbol)) for name, symbol in names.items()])


class Money(Processor):
    """Money, e.g. ``سبعة دولار وخمسون سنت`` -> ``$7.50`` and ``خمسة دينار كويتي`` -> ``د.ك5``.

    The tagger keeps the spoken amount, currency and minor-unit words; the
    verbalizer writes the currency symbol, digits and two-digit cents.
    """

    def __init__(self, cardinal):
        super().__init__(name="money", ordertype="itn", token_orders=TOKEN_ORDER)
        number = cardinal.graph
        major = "arabic/data/money/currency_major.tsv"
        local = "arabic/data/money/local_currency.tsv"
        # Names listed in both tables (ريال سعودي, درهم إماراتي) take the local symbol.
        self.currency = symbols(local, major).optimize()
        major_names = self.input_projection(symbols(major))
        local_names = self.input_projection(symbols(local))
        minor = [row[1] for name in ("singular", "plural")
                 for row in load_labels(get_abs_path("arabic/data/money/currency_minor_{}.tsv".format(name)))]
        minor_names = pynini.union(*[pynini.escape(name) for name in minor]).optimize()
        self.cents = (number @ (self.DIGIT**2 | insert("0") + self.DIGIT)).optimize()

        integer = self.tag_field("integer_part", number) + self.DELETE_EXTRA_SPACE
        cents = self.tag_field("fractional_part", self.cents) + self.DELETE_SPACE + insert(" ")
        cents += self.tag_field("currency_min", minor_names)
        joined = self.DELETE_SPACE + delete("و") + self.DELETE_ZERO_OR_ONE_SPACE + insert(" ") + cents
        graph = integer + self.tag_field("currency", major_names) + joined.ques
        graph |= integer + self.tag_field("currency", local_names)
        graph |= cents
        self.tagger = self.add_tokens(graph)

        currency = self.verbalize_field("currency", self.currency)
        amount = self.verbalize_field("integer_part", number, True)
        fractional = insert(".") + self.verbalize_field("fractional_part", self.cents, True)
        minor_unit = self.verbalize_field("currency_min", delete(minor_names), True)
        # Cents on their own are dollars, as in the original grammar.
        alone = insert("$0.") + self.verbalize_field("fractional_part", self.cents) + minor_unit
        self.verbalizer = self.delete_tokens(currency + amount + (fractional + minor_unit).ques | alone)
