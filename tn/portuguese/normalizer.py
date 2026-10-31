# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from tn.portuguese.rules import date, money
from tn.portuguese.rules.cardinal import Cardinal
from tn.portuguese.rules.date import Date
from tn.portuguese.rules.decimal import Decimal
from tn.portuguese.rules.electronic import Electronic
from tn.portuguese.rules.fraction import Fraction
from tn.portuguese.rules.measure import Measure
from tn.portuguese.rules.money import Money
from tn.portuguese.rules.ordinal import Ordinal
from tn.portuguese.rules.punctuation import Punctuation
from tn.portuguese.rules.telephone import Telephone
from tn.portuguese.rules.time import Time
from tn.portuguese.rules.whitelist import WhiteList
from tn.portuguese.rules.word import Word
from tn.processor import Processor, RuleSpec


TOKEN_ORDERS = {}
for _module in (date, money):
    TOKEN_ORDERS.update(_module.TOKEN_ORDER)


class Normalizer(Processor):

    def __init__(self, cache_dir=None, overwrite_cache=False):
        super().__init__(name="pt_normalizer", token_orders=TOKEN_ORDERS)
        self.build_fst("pt_tn", cache_dir, overwrite_cache, {})

    def build_tagger_and_verbalizer(self):
        cardinal = Cardinal()
        ordinal = Ordinal(cardinal)
        decimal = Decimal(cardinal)
        fraction = Fraction(cardinal, ordinal)
        punctuation = Punctuation()
        word = Word(punctuation)
        rules = (
            RuleSpec(WhiteList(), 1.01),
            RuleSpec(Date(cardinal), 1.1),
            RuleSpec(Time(cardinal), 1.1),
            RuleSpec(Measure(cardinal, decimal, fraction), 1.1),
            RuleSpec(fraction, 1.1),
            RuleSpec(decimal, 1.1),
            RuleSpec(ordinal, 1.1),
            RuleSpec(cardinal, 1.1),
            RuleSpec(Money(cardinal, decimal), 1.1),
            RuleSpec(Telephone(cardinal), 1.11),
            RuleSpec(Electronic(cardinal), 1.11),
            RuleSpec(word, 100, verbalize=False),
            RuleSpec(punctuation, verbalize=False),
        )
        classify = self.tagger_union(rules)
        punct = pynutil.add_weight(punctuation.tagger, 1.1)
        token = punct.star + classify + punct.star
        self.tagger = self.DELETE_SPACE + token + (self.DELETE_EXTRA_SPACE.ques + token).star + self.DELETE_SPACE
        self.tagger |= punct.plus
        self.tagger |= word.add_tokens(word.tag_field("value", self.SPACE.plus))
        self.verbalizer = self.join_tokens(self.verbalizer_union(rules), (word.verbalizer, punctuation.verbalizer))
