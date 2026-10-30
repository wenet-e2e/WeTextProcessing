# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

from tn.korean.rules import date, fraction, measure, money, time
from tn.korean.rules.cardinal import Cardinal
from tn.korean.rules.date import Date
from tn.korean.rules.decimal import Decimal
from tn.korean.rules.electronic import Electronic
from tn.korean.rules.fraction import Fraction
from tn.korean.rules.measure import Measure
from tn.korean.rules.money import Money
from tn.korean.rules.ordinal import Ordinal
from tn.korean.rules.punctuation import Punctuation
from tn.korean.rules.telephone import Telephone
from tn.korean.rules.time import Time
from tn.korean.rules.whitelist import WhiteList
from tn.korean.rules.word import Word
from tn.processor import Processor, RuleSpec

TOKEN_ORDERS = {}
for _module in (date, fraction, measure, money, time):
    TOKEN_ORDERS.update(_module.TOKEN_ORDER)


class Normalizer(Processor):

    def __init__(self, cache_dir=None, overwrite_cache=False):
        super().__init__(name="ko_normalizer", token_orders=TOKEN_ORDERS)
        self.build_fst("ko_tn", cache_dir, overwrite_cache, {})

    def build_tagger_and_verbalizer(self):
        cardinal = Cardinal()
        decimal = Decimal(cardinal)
        fraction = Fraction(cardinal)
        word = Word()
        rules = (
            RuleSpec(cardinal, 1.1),
            RuleSpec(Date(cardinal), 1.1),
            RuleSpec(Time(cardinal), 1.1),
            RuleSpec(fraction, 1.0),
            RuleSpec(Ordinal(cardinal), 1.1),
            RuleSpec(decimal, 1.05),
            RuleSpec(word, 100),
            RuleSpec(Money(cardinal), 1.1),
            RuleSpec(Measure(cardinal, decimal, fraction), 1.1),
            RuleSpec(Punctuation(), 1.0),
            RuleSpec(WhiteList(), 1.1),
            RuleSpec(Telephone(), 1.1),
            RuleSpec(Electronic(cardinal), 1.11),
        )
        classify = self.tagger_union(rules)
        separator = self.DELETE_EXTRA_SPACE.ques
        self.tagger = self.DELETE_SPACE + classify + (separator + classify).star + self.DELETE_SPACE
        self.tagger |= word.add_tokens(word.tag_field("value", self.SPACE.plus))
        # Spoken forms keep the trailing spaces some verbalizers emit (e.g. "이천이십사년 ").
        verbalize = self.verbalizer_union(rules)
        self.verbalizer = verbalize + (self.SPACE.plus + verbalize).star
