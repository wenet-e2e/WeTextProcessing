# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
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

from pynini.lib.pynutil import add_weight

from itn.vietnamese.rules import decimal, measure, money
from itn.vietnamese.rules.cardinal import Cardinal
from itn.vietnamese.rules.date import Date
from itn.vietnamese.rules.decimal import Decimal
from itn.vietnamese.rules.electronic import Electronic
from itn.vietnamese.rules.fraction import Fraction
from itn.vietnamese.rules.measure import Measure
from itn.vietnamese.rules.money import Money
from itn.vietnamese.rules.ordinal import Ordinal
from itn.vietnamese.rules.punctuation import Punctuation
from itn.vietnamese.rules.telephone import Telephone
from itn.vietnamese.rules.time import Time
from itn.vietnamese.rules.whitelist import WhiteList
from itn.vietnamese.rules.word import Word
from tn.processor import Processor, RuleSpec


TOKEN_ORDERS = {}
for _module in (decimal, measure, money):
    TOKEN_ORDERS.update(_module.TOKEN_ORDER)


class InverseNormalizer(Processor):

    def __init__(self, cache_dir=None, overwrite_cache=False):
        super().__init__(name="vi_inverse_normalizer", ordertype="itn", token_orders=TOKEN_ORDERS)
        self.build_fst("vi_itn", cache_dir, overwrite_cache, {})

    def build_tagger_and_verbalizer(self):
        cardinal = Cardinal()
        decimal = Decimal(cardinal)
        punctuation = Punctuation()
        word = Word()
        rules = (
            RuleSpec(WhiteList(), 1.01),
            RuleSpec(Time(), 1.09),
            RuleSpec(Date(cardinal), 1.1),
            RuleSpec(decimal, 1.1),
            RuleSpec(Measure(cardinal, decimal), 1.1),
            RuleSpec(cardinal, 1.1),
            RuleSpec(Ordinal(), 1.1),
            RuleSpec(Fraction(cardinal), 1.1),
            RuleSpec(Money(cardinal, decimal), 1.1),
            RuleSpec(Telephone(cardinal), 1.1),
            RuleSpec(Electronic(cardinal), 1.11),
            RuleSpec(word, 100),
            RuleSpec(punctuation),
        )
        classify = self.tagger_union(rules)
        punct = add_weight(punctuation.tagger, 1.1)
        token = punct.star + classify + punct.star
        self.tagger = self.DELETE_SPACE + token + (self.DELETE_EXTRA_SPACE + token).star + self.DELETE_SPACE
        self.tagger |= word.add_tokens(word.tag_field("value", self.SPACE.plus))
        verbalize = self.verbalizer_union(rules)
        self.verbalizer = verbalize + (self.INSERT_SPACE + verbalize).star
