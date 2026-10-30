# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pynini.lib import pynutil

from tn.german.rules import date, measure, money, time
from tn.german.rules.cardinal import Cardinal
from tn.german.rules.date import Date
from tn.german.rules.decimal import Decimal
from tn.german.rules.electronic import Electronic
from tn.german.rules.fraction import Fraction
from tn.german.rules.measure import Measure
from tn.german.rules.money import Money
from tn.german.rules.ordinal import Ordinal
from tn.german.rules.punctuation import Punctuation
from tn.german.rules.telephone import Telephone
from tn.german.rules.time import Time
from tn.german.rules.whitelist import WhiteList
from tn.german.rules.word import Word
from tn.processor import Processor, RuleSpec


TOKEN_ORDERS = {}
for _module in (date, measure, money, time):
    TOKEN_ORDERS.update(_module.TOKEN_ORDER)


class Normalizer(Processor):
    def __init__(self, cache_dir=None, overwrite_cache=False, deterministic=True):
        super().__init__(name="de_normalizer", token_orders=TOKEN_ORDERS)
        self.deterministic = deterministic
        self.build_fst("de_tn", cache_dir, overwrite_cache, {"deterministic": deterministic})

    def build_tagger_and_verbalizer(self):
        cardinal = Cardinal(self.deterministic)
        ordinal = Ordinal(cardinal, self.deterministic)
        decimal = Decimal(cardinal, self.deterministic)
        fraction = Fraction(cardinal, ordinal, self.deterministic)
        punctuation = Punctuation(self.deterministic)
        word = Word(punctuation, self.deterministic)
        rules = (
            RuleSpec(WhiteList(self.deterministic), 1.01),
            RuleSpec(Time(cardinal, self.deterministic), 1.1),
            RuleSpec(Measure(cardinal, decimal, fraction, self.deterministic), 1.1),
            RuleSpec(cardinal, 1.1),
            RuleSpec(fraction, 1.1),
            RuleSpec(Date(cardinal, ordinal, self.deterministic), 1.1),
            RuleSpec(ordinal, 1.1),
            RuleSpec(decimal, 1.1),
            RuleSpec(Money(cardinal, decimal, self.deterministic), 1.1),
            RuleSpec(Telephone(cardinal, self.deterministic), 1.1),
            RuleSpec(Electronic(self.deterministic), 1.11),
            RuleSpec(word, 100),
            RuleSpec(punctuation),
        )
        classify = self.tagger_union(rules)
        punct = pynutil.add_weight(punctuation.tagger, 1.1)
        token = punct.star + classify + punct.star
        self.tagger = self.DELETE_SPACE + token + (self.DELETE_EXTRA_SPACE.ques + token).star + self.DELETE_SPACE
        self.tagger |= punct.plus
        self.tagger |= word.add_tokens(word.tag_field("value", self.SPACE.plus))
        verbalize = self.verbalizer_union(rules)
        self.verbalizer = verbalize + (self.INSERT_SPACE + verbalize).star
        self.verbalizer @= self.build_rule(pynini.cross("\u00a0", " "))
