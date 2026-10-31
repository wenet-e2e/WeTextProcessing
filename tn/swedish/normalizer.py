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

from pynini.lib import pynutil

from tn.processor import Processor, RuleSpec
from tn.swedish.rules import date, measure, money, time
from tn.swedish.rules.cardinal import Cardinal
from tn.swedish.rules.date import Date
from tn.swedish.rules.decimal import Decimal
from tn.swedish.rules.electronic import Electronic
from tn.swedish.rules.fraction import Fraction
from tn.swedish.rules.measure import Measure
from tn.swedish.rules.money import Money
from tn.swedish.rules.ordinal import Ordinal
from tn.swedish.rules.punctuation import Punctuation
from tn.swedish.rules.telephone import Telephone
from tn.swedish.rules.time import Time
from tn.swedish.rules.whitelist import WhiteList
from tn.swedish.rules.word import Word

TOKEN_ORDERS = {}
for _module in (date, measure, money, time):
    TOKEN_ORDERS.update(_module.TOKEN_ORDER)


class Normalizer(Processor):
    def __init__(self, cache_dir=None, overwrite_cache=False, deterministic=True):
        super().__init__(name="sv_normalizer", token_orders=TOKEN_ORDERS)
        self.deterministic = deterministic
        self.build_fst("sv_tn", cache_dir, overwrite_cache, {"deterministic": deterministic})

    def build_tagger_and_verbalizer(self):
        deterministic = self.deterministic
        cardinal = Cardinal(deterministic)
        ordinal = Ordinal(cardinal, deterministic)
        decimal = Decimal(cardinal, deterministic)
        fraction = Fraction(cardinal, ordinal, deterministic)
        punctuation = Punctuation(deterministic)
        word = Word(punctuation, deterministic)
        rules = (
            RuleSpec(WhiteList(deterministic), 1.01),
            RuleSpec(Time(cardinal, deterministic), 1.1),
            RuleSpec(Date(cardinal, ordinal, deterministic), 1.09),
            RuleSpec(decimal, 1.1),
            RuleSpec(Measure(cardinal, decimal, fraction, deterministic), 1.1),
            RuleSpec(cardinal, 1.1),
            RuleSpec(ordinal, 1.1),
            RuleSpec(Money(cardinal, decimal, deterministic), 1.1),
            RuleSpec(Telephone(cardinal, deterministic), 1.1),
            RuleSpec(Electronic(deterministic), 1.1),
            RuleSpec(fraction, 1.1),
            RuleSpec(word, 100, verbalize=False),
            RuleSpec(punctuation, verbalize=False),
        )
        classify = self.tagger_union(rules)
        punct = pynutil.add_weight(punctuation.tagger, 2.1)
        separator = self.DELETE_EXTRA_SPACE | punct.plus
        tokens = punct.star + classify + (separator + classify).star + punct.star
        self.tagger = self.DELETE_SPACE + tokens + self.DELETE_SPACE
        self.tagger |= punct.plus
        self.tagger |= word.add_tokens(word.tag_field("value", self.SPACE.plus))
        self.verbalizer = self.join_tokens(self.verbalizer_union(rules), (word.verbalizer, punctuation.verbalizer))
