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
from tn.spanish.rules import date, electronic, measure, money, telephone, time
from tn.spanish.rules.cardinal import Cardinal
from tn.spanish.rules.date import Date
from tn.spanish.rules.decimal import Decimal
from tn.spanish.rules.electronic import Electronic
from tn.spanish.rules.fraction import Fraction
from tn.spanish.rules.measure import Measure
from tn.spanish.rules.money import Money
from tn.spanish.rules.ordinal import Ordinal
from tn.spanish.rules.punctuation import Punctuation
from tn.spanish.rules.telephone import Telephone
from tn.spanish.rules.time import Time
from tn.spanish.rules.whitelist import WhiteList
from tn.spanish.rules.word import Word

TOKEN_ORDERS = {}
for _module in (date, electronic, measure, money, telephone, time):
    TOKEN_ORDERS.update(_module.TOKEN_ORDER)


class Normalizer(Processor):

    def __init__(self, cache_dir=None, overwrite_cache=False, deterministic=True):
        super().__init__(name="es_normalizer", token_orders=TOKEN_ORDERS)
        self.deterministic = deterministic
        self.build_fst("es_tn", cache_dir, overwrite_cache, {"deterministic": deterministic})

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
            RuleSpec(Time(cardinal, deterministic), 1.09),
            RuleSpec(Measure(cardinal, decimal, fraction, deterministic), 1.08),
            RuleSpec(cardinal, 1.1),
            RuleSpec(fraction, 1.09),
            RuleSpec(Date(cardinal, deterministic), 1.1),
            RuleSpec(ordinal, 1.1),
            RuleSpec(decimal, 1.1),
            RuleSpec(Money(cardinal, decimal, deterministic), 1.09),
            RuleSpec(Telephone(deterministic), 1.11),
            RuleSpec(Electronic(deterministic), 1.11),
            RuleSpec(word, 200),
            RuleSpec(punctuation),
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
        verbalize = self.verbalizer_union(rules)
        self.verbalizer = verbalize + (self.INSERT_SPACE + verbalize).star
