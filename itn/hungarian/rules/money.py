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

from itn.hungarian.rules.utils import DIGIT, data_labels, inflect_abbreviation, naive_inflector
from tn.processor import Processor

TOKEN_ORDER = {}


def currencies():
    """Returns ``(spoken name, written symbol, minor unit)`` with the first listed symbol per name."""

    minors = dict((row[0], row[1]) for row in data_labels("money/currency_minor.tsv"))
    seen = {}
    for symbol, name in data_labels("money/currency.tsv"):
        seen.setdefault(name, (symbol, minors[symbol]))
    return [(name, symbol, minor) for name, (symbol, minor) in seen.items()]


class Money(Processor):
    """Money, e.g. ``egy dollár egy cent`` -> ``$1,01``, ``egy milliárd angol fontot`` -> ``1 mrd GBP-t``.

    Raw spoken amount fields, ``currency`` and optional ``minor`` amount with
    its ``minor_currency``. Symbols (``$``, ``£``) are written before the
    amount, letter codes (``GBP``, ``Ft``) after it with their case endings.
    """

    def __init__(self, cardinal, decimal):
        super().__init__("money", ordertype="itn", token_orders=TOKEN_ORDER)
        cents = (pynini.invert(cardinal.spoken) @ pynini.closure(DIGIT, 1, 2)
                 @ (insert("0") + DIGIT | DIGIT ** 2)).optimize()
        integer_t = self.tag_field("integer_part", cardinal.graph)
        amount_t = integer_t | decimal.unsigned_tagger
        integer_v = self.verbalize_field("integer_part", cardinal.graph)
        amount_v = integer_v | decimal.unsigned_verbalizer

        taggers, verbalizers = [], []
        for name, symbol, minor in currencies():
            symbol_written = pynini.escape(symbol)
            prefix = re.search(r"[^A-Za-z]", symbol) is not None
            forms = [(symbol, name)]
            if not prefix:
                inflected = inflect_abbreviation(symbol, name) if re.match("^[A-Z]{3}$", symbol) else []
                forms += inflected + naive_inflector(symbol, name)
            written = {}
            for abbr, spoken in forms:
                written.setdefault(spoken, abbr)
            currency = pynini.string_map([(spoken, abbr) for spoken, abbr in written.items()]).optimize()
            bare = pynini.cross(pynini.escape(name), symbol_written)
            minor_t = (pynini.accep(" ") + self.tag_field("minor", cents) + pynini.accep(" ")
                       + self.tag_field("minor_currency", pynini.accep(minor)))
            minor_v = (pynini.cross(" ", ",") + self.verbalize_field("minor", cents)
                       + self.verbalize_field("minor_currency", delete(minor), leading_space=True))
            taggers.append(amount_t + pynini.accep(" ") + self.tag_field("currency", currency))
            taggers.append(integer_t + pynini.accep(" ") + self.tag_field("currency", bare) + minor_t)
            if prefix:
                delete_currency = self.verbalize_field("currency", delete(pynini.escape(name)), leading_space=True)
                verbalizers.append(insert(symbol_written) + amount_v + delete_currency)
                verbalizers.append(insert(symbol_written) + integer_v + delete_currency + minor_v)
            else:
                currency_v = pynini.accep(" ") + self.verbalize_field("currency", currency)
                verbalizers.append(amount_v + currency_v)
                bare_v = self.verbalize_field("currency", delete(pynini.escape(name)), leading_space=True)
                verbalizers.append(integer_v + bare_v + minor_v + insert(" " + symbol_written))
        self.tagger = self.add_tokens(pynini.union(*taggers))
        self.verbalizer = self.delete_tokens(pynini.union(*verbalizers))
