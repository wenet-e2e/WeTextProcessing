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

from pynini.lib import pynutil

from tn.hungarian.rules import date, electronic, money, telephone, time
from tn.hungarian.rules.cardinal import Cardinal
from tn.hungarian.rules.date import Date
from tn.hungarian.rules.decimal import Decimal
from tn.hungarian.rules.electronic import Electronic
from tn.hungarian.rules.fraction import Fraction
from tn.hungarian.rules.measure import Measure
from tn.hungarian.rules.money import Money
from tn.hungarian.rules.ordinal import Ordinal
from tn.hungarian.rules.punctuation import Punctuation
from tn.hungarian.rules.telephone import Telephone
from tn.hungarian.rules.time import Time
from tn.hungarian.rules.whitelist import WhiteList
from tn.hungarian.rules.word import Word
from tn.processor import Processor, RuleSpec

TOKEN_ORDERS = {}
for _module in (date, electronic, money, telephone, time):
    TOKEN_ORDERS.update(_module.TOKEN_ORDER)


class Normalizer(Processor):

    def __init__(self, cache_dir=None, overwrite_cache=False):
        super().__init__(name="hu_normalizer", token_orders=TOKEN_ORDERS)
        self.build_fst("hu_tn", cache_dir, overwrite_cache, {})

    def build_tagger_and_verbalizer(self):
        cardinal = Cardinal()
        ordinal = Ordinal(cardinal)
        decimal = Decimal(cardinal)
        fraction = Fraction(cardinal, ordinal)
        punctuation = Punctuation()
        word = Word(punctuation)
        rules = (
            RuleSpec(WhiteList(), 1.01),
            RuleSpec(Time(cardinal), 1.09),
            RuleSpec(Measure(cardinal, decimal, fraction), 1.08),
            RuleSpec(cardinal, 1.1),
            RuleSpec(fraction, 1.09),
            RuleSpec(Date(cardinal), 1.1),
            RuleSpec(ordinal, 1.1),
            RuleSpec(decimal, 1.1),
            RuleSpec(Money(cardinal, decimal), 1.1),
            # Slightly above cardinal: "1 034 068" is also a valid local number.
            RuleSpec(Telephone(cardinal), 1.105),
            RuleSpec(Electronic(), 1.11),
            RuleSpec(word, 200, verbalize=False),
            RuleSpec(punctuation, verbalize=False),
        )
        classify = self.tagger_union(rules)
        # Classified tokens are separated by spaces or punctuation; punctuation
        # may stick to either side of a token.
        space = self.DELETE_EXTRA_SPACE
        punct = pynutil.add_weight(punctuation.tagger, 2.1)
        punct = (punct + (space.ques + punct).star).optimize()
        separator = space | space.ques + punct + space.ques
        tokens = (punct + space.ques).ques + classify + (separator + classify).star + (space.ques + punct).ques
        self.tagger = self.DELETE_SPACE + (tokens | punct) + self.DELETE_SPACE
        self.tagger |= word.add_tokens(word.tag_field("value", self.SPACE.plus))
        self.verbalizer = self.join_tokens(
            self.verbalizer_union(rules), (word.verbalizer, punctuation.verbalizer),
            opening=self.OPENING_MARKS + ("-",), closing=self.CLOSING_MARKS + ("-",))
