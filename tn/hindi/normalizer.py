# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from pynini.lib.pynutil import add_weight, delete

from tn.hindi.rules import (address, cardinal, date, decimal, electronic, fraction, measure, money, ordinal,
                            punctuation, roman, serial, telephone, time, whitelist, word)
from tn.hindi.rules.address import Address
from tn.hindi.rules.cardinal import Cardinal
from tn.hindi.rules.date import Date
from tn.hindi.rules.decimal import Decimal
from tn.hindi.rules.electronic import Electronic
from tn.hindi.rules.fraction import Fraction
from tn.hindi.rules.measure import Measure
from tn.hindi.rules.money import Money
from tn.hindi.rules.ordinal import Ordinal
from tn.hindi.rules.punctuation import Punctuation
from tn.hindi.rules.roman import Roman
from tn.hindi.rules.serial import Serial
from tn.hindi.rules.telephone import Telephone
from tn.hindi.rules.time import Time
from tn.hindi.rules.whitelist import WhiteList
from tn.hindi.rules.word import Word
from tn.processor import Processor, RuleSpec

TOKEN_ORDERS = {}
for _module in (address, cardinal, date, decimal, electronic, fraction, measure, money, ordinal,
                punctuation, roman, serial, telephone, time, whitelist, word):
    TOKEN_ORDERS.update(_module.TOKEN_ORDER)

# NeMo's classifier weight for punctuation tokens.
PUNCT_WEIGHT = 2.1


class Normalizer(Processor):

    def __init__(self, cache_dir=None, overwrite_cache=False):
        super().__init__(name="hi_normalizer", token_orders=TOKEN_ORDERS)
        self.build_fst("hi_tn", cache_dir, overwrite_cache, {})

    def build_tagger_and_verbalizer(self):
        cardinal = Cardinal()
        decimal = Decimal(cardinal)
        ordinal = Ordinal(cardinal)
        serial = Serial(cardinal)
        punctuation = Punctuation()
        word = Word(punctuation)
        rules = (
            RuleSpec(WhiteList(), 1.01),
            RuleSpec(cardinal, 1.1),
            RuleSpec(decimal, 1.1),
            RuleSpec(Fraction(cardinal), 1.05),
            RuleSpec(Date(cardinal), 1.1),
            RuleSpec(Time(cardinal), 1.1),
            RuleSpec(Measure(cardinal, decimal), 1.1),
            RuleSpec(Address(cardinal, ordinal, serial), 1.1),
            RuleSpec(Money(cardinal), 1.1),
            RuleSpec(Telephone(), 1.1),
            RuleSpec(ordinal, 1.1),
            RuleSpec(Electronic(), 1.1),
            RuleSpec(serial, 1.11),
            RuleSpec(Roman(), 1.1),
            RuleSpec(word, 100),
            RuleSpec(punctuation),
        )
        classify = self.tagger_union(rules)

        def punct(graph):
            return add_weight(graph, PUNCT_WEIGHT)

        # Punctuation values carry their surrounding spaces, so tokens either meet at a
        # punctuation token or are separated by (deleted) whitespace.
        connector = delete(self.SPACE.plus) | punct(punctuation.middle)
        body = classify + (connector + classify).star
        tagger = punct(punctuation.leading).ques + body + punct(punctuation.trailing).ques
        tagger |= punct(punctuation.only)
        self.tagger = self.DELETE_SPACE + tagger + self.DELETE_SPACE
        self.tagger |= word.add_tokens(word.tag_field("value", self.SPACE.plus))

        verbalize = self.verbalizer_union(rules[:-1])
        punct_verbalize = punctuation.verbalizer
        body = verbalize + ((self.INSERT_SPACE | punct_verbalize) + verbalize).star
        self.verbalizer = punct_verbalize.ques + body + punct_verbalize.ques | punct_verbalize
