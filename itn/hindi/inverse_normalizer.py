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

from itn.hindi.rules import (cardinal, date, decimal, fraction, measure, money, ordinal, punctuation, telephone, time,
                             whitelist, word)
from itn.hindi.rules.cardinal import Cardinal
from itn.hindi.rules.date import Date
from itn.hindi.rules.decimal import Decimal
from itn.hindi.rules.fraction import Fraction
from itn.hindi.rules.measure import Measure
from itn.hindi.rules.money import Money
from itn.hindi.rules.ordinal import Ordinal
from itn.hindi.rules.punctuation import Punctuation
from itn.hindi.rules.telephone import Telephone
from itn.hindi.rules.time import Time
from itn.hindi.rules.whitelist import Whitelist
from itn.hindi.rules.word import Word
from tn.processor import Processor, RuleSpec

TOKEN_ORDERS = {}
for _module in (cardinal, date, decimal, fraction, measure, money, ordinal, punctuation, telephone, time, whitelist,
                word):
    TOKEN_ORDERS.update(_module.TOKEN_ORDER)

# Classifier weight of a punctuation mark glued to a token.
PUNCT_WEIGHT = 1.1


class InverseNormalizer(Processor):

    def __init__(self, cache_dir=None, overwrite_cache=False):
        super().__init__(name="hi_inverse_normalizer", ordertype="itn", token_orders=TOKEN_ORDERS)
        self.build_fst("hi_itn", cache_dir, overwrite_cache, {})

    def build_tagger_and_verbalizer(self):
        cardinal = Cardinal()
        decimal = Decimal(cardinal)
        punctuation = Punctuation()
        word = Word()
        rules = (
            RuleSpec(cardinal, 1.1),
            RuleSpec(Ordinal(cardinal), 1.1),
            RuleSpec(decimal, 1.1),
            RuleSpec(Fraction(cardinal), 1.1),
            RuleSpec(Date(cardinal), 1.1),
            RuleSpec(Time(cardinal), 1.1),
            RuleSpec(Measure(cardinal, decimal), 1.1),
            RuleSpec(Money(cardinal, decimal), 1.1),
            RuleSpec(Telephone(), 1.1),
            RuleSpec(word, 100),
            RuleSpec(Whitelist(), 1.01),
            RuleSpec(punctuation),
        )
        classify = self.tagger_union(rules)
        # Punctuation marks are separate tokens only when glued to a token ("याहू!" stays a word,
        # because a word costs less than a word plus a punctuation token).
        punct = add_weight(punctuation.tagger, PUNCT_WEIGHT)
        token = punct.star + classify + punct.star
        tagger = token + (delete(self.SPACE.plus) + token).star
        self.tagger = self.DELETE_SPACE + tagger + self.DELETE_SPACE
        self.tagger |= word.add_tokens(word.tag_field("value", self.SPACE.plus))

        verbalize = self.verbalizer_union(rules)
        self.verbalizer = verbalize + (self.INSERT_SPACE + verbalize).star
