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

from itn.italian.rules.utils import load_rows, string_map
from tn.processor import Processor

# The verbalizer writes the amount before the currency symbol.
TOKEN_ORDER = {"money": ["integer_part", "fractional_part", "currency", "minor"]}


class Money(Processor):
    """Amounts of money, e.g. ``due rubli un copeca`` -> ``2,01 ₽``.

    The tagger keeps the spoken amount, the currency name and the minor
    unit name (an optional ``e`` connector is dropped); the verbalizer writes
    ``<amount> <symbol>`` with two-digit minor units. Minor units alone
    (``cinquanta centesimi``) are written as ``0,50`` of the main currency.
    """

    def __init__(self, cardinal, decimal):
        super().__init__("money", token_orders=TOKEN_ORDER)
        rows = load_rows("italian/data/money/currency.tsv")
        majors = string_map([(row[0], spoken) for row in rows for spoken in row[1:3]])
        minor_symbols = {}
        for row in rows:
            for spoken in row[3:5]:
                if spoken:
                    minor_symbols.setdefault(spoken, row[0])
        minors_alone = string_map([(symbol, spoken) for spoken, symbol in minor_symbols.items()])
        cents = (cardinal.two | pynini.cross(pynini.union("un", "una"), "01")).optimize()

        integer_tag = self.tag_field("integer_part", cardinal.counted)
        space = pynini.accep(" ")
        # The minor unit must belong to the named currency.
        named = []
        for row in rows:
            major = self.tag_field("currency", pynini.union(*row[1:3]))
            names = [spoken for spoken in row[3:5] if spoken]
            if names:
                minor = space + self.tag_field("minor", pynini.union(*names))
                fraction = (delete(" e") | "") + space + self.tag_field("fractional_part", cents) + minor
                major += fraction.ques
            named.append(major)
        any_major = self.tag_field("currency", majors)
        minor_alone_tag = self.tag_field("fractional_part", cents) + space + self.tag_field("minor", minors_alone)
        branches = [
            integer_tag + space + pynini.union(*named),
            decimal.fraction_tagger + space + any_major,
            minor_alone_tag,
        ]
        self.tagger = self.add_tokens(pynini.union(*branches))

        integer_verb = self.verbalize_field("integer_part", cardinal.counted)
        cents_verb = delete(" ") + insert(",") + self.verbalize_field("fractional_part", cents)
        currency_verb = delete(" ") + insert(" ") + self.verbalize_field("currency", majors)
        minor_verb = delete(" ") + self.verbalize_field("minor", delete(pynini.union(*minor_symbols)))
        minor_alone = delete(" ") + insert(" ") + self.verbalize_field("minor", minors_alone)
        verbalizer = pynini.union(
            integer_verb + currency_verb,
            decimal.fraction_verbalizer + currency_verb,
            integer_verb + cents_verb + currency_verb + minor_verb,
            insert("0,") + self.verbalize_field("fractional_part", cents) + minor_alone,
        )
        self.verbalizer = self.delete_tokens(verbalizer)
