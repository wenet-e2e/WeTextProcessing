# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from tn.italian.rules import money
from tn.italian.rules.cardinal import Cardinal
from tn.italian.rules.decimal import Decimal
from tn.italian.rules.electronic import Electronic
from tn.italian.rules.measure import Measure
from tn.italian.rules.money import Money
from tn.italian.rules.punctuation import Punctuation
from tn.italian.rules.time import Time
from tn.italian.rules.whitelist import WhiteList
from tn.italian.rules.word import Word
from tn.processor import Processor, RuleSpec

TOKEN_ORDERS = dict(money.TOKEN_ORDER)


class Normalizer(Processor):

    def __init__(self, cache_dir=None, overwrite_cache=False):
        super().__init__(name="it_normalizer", token_orders=TOKEN_ORDERS)
        self.build_fst("it_tn", cache_dir, overwrite_cache, {})

    def build_tagger_and_verbalizer(self):
        cardinal = Cardinal()
        decimal = Decimal(cardinal)
        punctuation = Punctuation()
        word = Word()
        rules = (
            RuleSpec(WhiteList(), 1.01),
            RuleSpec(Time(cardinal), 1.1),
            RuleSpec(decimal, 1.1),
            RuleSpec(Measure(cardinal, decimal), 0.9),
            RuleSpec(cardinal, 1.1),
            RuleSpec(Money(cardinal, decimal), 1.1),
            RuleSpec(Electronic(cardinal), 1.1),
            RuleSpec(word, 100, verbalize=False),
            RuleSpec(punctuation, verbalize=False),
        )
        classify = self.tagger_union(rules)
        punct = add_weight(punctuation.tagger, 1.1)
        token = punct.star + classify + punct.star
        self.tagger = self.DELETE_SPACE + token + (self.DELETE_EXTRA_SPACE + token).star + self.DELETE_SPACE
        self.tagger |= word.add_tokens(word.tag_field("value", self.SPACE.plus))
        self.verbalizer = self.join_tokens(self.verbalizer_union(rules), (word.verbalizer, punctuation.verbalizer))
