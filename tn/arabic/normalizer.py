# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from tn.arabic.rules import measure, money
from tn.arabic.rules.cardinal import Cardinal
from tn.arabic.rules.decimal import Decimal
from tn.arabic.rules.fraction import Fraction
from tn.arabic.rules.measure import Measure
from tn.arabic.rules.money import Money
from tn.arabic.rules.punctuation import Punctuation
from tn.arabic.rules.word import Word
from tn.processor import Processor, RuleSpec

TOKEN_ORDERS = {}
for _module in (measure, money):
    TOKEN_ORDERS.update(_module.TOKEN_ORDER)


class Normalizer(Processor):

    def __init__(self, cache_dir=None, overwrite_cache=False):
        super().__init__(name="ar_normalizer", token_orders=TOKEN_ORDERS)
        self.build_fst("ar_tn", cache_dir, overwrite_cache, {})

    def build_tagger_and_verbalizer(self):
        cardinal = Cardinal()
        decimal = Decimal(cardinal)
        fraction = Fraction(cardinal)
        punctuation = Punctuation()
        word = Word(punctuation)
        rules = (
            RuleSpec(cardinal, 1.1),
            RuleSpec(decimal, 1.1),
            RuleSpec(fraction, 1.0),
            RuleSpec(Money(cardinal), 1.0),
            RuleSpec(Measure(cardinal, decimal, fraction), 1.0),
            RuleSpec(word, 100),
            RuleSpec(punctuation),
        )
        classify = self.tagger_union(rules)
        punct = pynutil.add_weight(punctuation.tagger, 1.1)
        token = punct.star + classify + punct.star
        separator = pynutil.add_weight(self.DELETE_EXTRA_SPACE, 1.1)
        self.tagger = self.DELETE_SPACE + token + (separator + token).star + self.DELETE_SPACE
        self.tagger |= punct.plus
        self.tagger |= word.add_tokens(word.tag_field("value", self.SPACE.plus))
        verbalize = self.verbalizer_union(rules)
        self.verbalizer = verbalize + (self.INSERT_SPACE + verbalize).star
