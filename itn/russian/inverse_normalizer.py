# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from itn.russian.rules import time
from itn.russian.rules.cardinal import Cardinal
from itn.russian.rules.date import Date
from itn.russian.rules.decimal import Decimal
from itn.russian.rules.electronic import Electronic
from itn.russian.rules.measure import Measure
from itn.russian.rules.money import Money
from itn.russian.rules.ordinal import Ordinal
from itn.russian.rules.punctuation import Punctuation
from itn.russian.rules.telephone import Telephone
from itn.russian.rules.time import Time
from itn.russian.rules.whitelist import WhiteList
from itn.russian.rules.word import Word
from tn.processor import Processor, RuleSpec


TOKEN_ORDERS = {}
for _module in (time,):
    TOKEN_ORDERS.update(_module.TOKEN_ORDER)


class InverseNormalizer(Processor):
    def __init__(self, cache_dir=None, overwrite_cache=False):
        super().__init__(name="ru_inverse_normalizer", ordertype="itn", token_orders=TOKEN_ORDERS)
        self.build_fst("ru_itn", cache_dir, overwrite_cache, {})

    def build_tagger_and_verbalizer(self):
        cardinal = Cardinal()
        decimal = Decimal(cardinal)
        punctuation = Punctuation()
        word = Word(punctuation)
        rules = (
            RuleSpec(WhiteList(), 1.01),
            RuleSpec(Time(), 1.1),
            RuleSpec(Date(), 1.09),
            RuleSpec(decimal, 1.1),
            RuleSpec(Measure(cardinal, decimal), 1.1),
            RuleSpec(Ordinal(), 1.1),
            RuleSpec(Money(cardinal, decimal), 1.1),
            RuleSpec(Telephone(), 1.1),
            RuleSpec(Electronic(), 1.1),
            RuleSpec(cardinal, 1.1),
            RuleSpec(word, 100),
            RuleSpec(punctuation, 1.1),
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
