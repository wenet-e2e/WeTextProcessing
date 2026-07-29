# Copyright (c) 2022 Zhendong Peng (pzd17@tsinghua.org.cn)
# Copyright (c) 2024 Xingchen Song (sxc19@tsinghua.org.cn)
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

from tn.english.rules.cardinal import Cardinal
from tn.english.rules.date import Date
from tn.english.rules.decimal import Decimal
from tn.english.rules.electronic import Electronic
from tn.english.rules.fraction import Fraction
from tn.english.rules.measure import Measure
from tn.english.rules.money import Money
from tn.english.rules.ordinal import Ordinal
from tn.english.rules.punctuation import Punctuation
from tn.english.rules.range import Range
from tn.english.rules.serial import Serial
from tn.english.rules.telephone import Telephone
from tn.english.rules.time import Time
from tn.english.rules.whitelist import WhiteList
from tn.english.rules.word import Word
from tn.processor import Processor, RuleSpec

TOKEN_ORDERS = {
    "date": ["preserve_order", "text", "day", "month", "year"],
    "money": ["integer_part", "fractional_part", "quantity", "currency_maj"],
}


class Normalizer(Processor):

    def __init__(self, cache_dir=None, overwrite_cache=False):
        super().__init__(name="en_normalizer", ordertype="en_tn", token_orders=TOKEN_ORDERS)
        self.build_fst("en_tn", cache_dir, overwrite_cache, {})

    def build_tagger_and_verbalizer(self):
        cardinal = Cardinal()
        ordinal = Ordinal(cardinal=cardinal)
        decimal = Decimal(cardinal=cardinal)
        fraction = Fraction(cardinal=cardinal, ordinal=ordinal)
        punctuation = Punctuation()
        date = Date(cardinal=cardinal, ordinal=ordinal)
        time = Time(cardinal=cardinal)
        measure = Measure(cardinal=cardinal, decimal=decimal, fraction=fraction, ordinal=ordinal)
        money = Money(cardinal=cardinal, decimal=decimal)
        telephone = Telephone()
        electronic = Electronic(cardinal=cardinal)
        serial = Serial(cardinal=cardinal, ordinal=ordinal)
        word = Word(punctuation=punctuation)
        whitelist = WhiteList()
        rang = Range(date=date, time=time)

        rules = (
            RuleSpec(cardinal, 1.0),
            RuleSpec(ordinal, 1.0),
            RuleSpec(word, 100),
            RuleSpec(date, 0.99),
            RuleSpec(decimal, 1.0),
            RuleSpec(fraction, 0.99),
            RuleSpec(time, 1.0),
            RuleSpec(measure, 1.0),
            RuleSpec(money, 1.0),
            RuleSpec(telephone, 1.0),
            RuleSpec(electronic, 1.0),
            RuleSpec(serial, 1.01),
            RuleSpec(whitelist, 1.0),
            RuleSpec(rang, 1.0),
        )
        classify = self.tagger_union(rules)

        punct = pynutil.add_weight(punctuation.tagger, 2.00)
        token = pynini.closure(punct) + classify + pynini.closure(punct)
        separator = pynutil.delete(self.SPACE) | punct
        graph = (self.DELETE_SPACE + token + pynini.closure(separator + token) + self.DELETE_SPACE) | punct
        self.tagger = graph.optimize() @ self.build_rule(pynutil.delete(" "), r="[EOS]")

        classify = self.verbalizer_union(rules)
        punct = punctuation.verbalizer.optimize()
        # Punct tokens carry surrounding spacing in their values (the tagger's
        # add_weight(accep(" "), -1.0).star absorbs spaces around punctuation).
        # So punct tokens handle their own spacing and don't need INSERT_SPACE.
        # Only classify tokens need INSERT_SPACE for inter-word spacing.
        verbalizer = (classify + (punct.plus | pynutil.add_weight(self.INSERT_SPACE, 0.0001))
                      | punct + (punct.plus | self.DELETE_SPACE)).star
        self.verbalizer = verbalizer @ self.build_rule(pynutil.delete(" "), r="[EOS]")
