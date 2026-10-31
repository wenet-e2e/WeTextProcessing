# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from itn.arabic.rules import money
from itn.arabic.rules.cardinal import Cardinal
from itn.arabic.rules.decimal import Decimal
from itn.arabic.rules.fraction import Fraction
from itn.arabic.rules.measure import Measure
from itn.arabic.rules.money import Money
from itn.arabic.rules.punctuation import Punctuation
from itn.arabic.rules.word import Word
from tn.processor import Processor, RuleSpec

TOKEN_ORDERS = {}
for _module in (money, ):
    TOKEN_ORDERS.update(_module.TOKEN_ORDER)


class InverseNormalizer(Processor):

    def __init__(self, cache_dir=None, overwrite_cache=False):
        super().__init__(name="ar_inverse_normalizer", ordertype="itn", token_orders=TOKEN_ORDERS)
        self.build_fst("ar_itn", cache_dir, overwrite_cache, {})

    def build_tagger_and_verbalizer(self):
        cardinal = Cardinal()
        decimal = Decimal(cardinal)
        fraction = Fraction(cardinal)
        punctuation = Punctuation()
        word = Word()
        rules = (
            RuleSpec(cardinal, 1.1),
            RuleSpec(decimal, 1.1),
            RuleSpec(fraction, 1.1),
            RuleSpec(Money(cardinal), 1.1),
            RuleSpec(Measure(cardinal, decimal, fraction), 1.1),
            RuleSpec(word, 100),
            RuleSpec(punctuation),
        )
        classify = self.tagger_union(rules)
        punct = pynutil.add_weight(punctuation.tagger, 1.1)
        token = punct.star + classify + punct.star
        self.tagger = self.DELETE_SPACE + token + (self.DELETE_EXTRA_SPACE + token).star + self.DELETE_SPACE
        self.tagger |= word.add_tokens(word.tag_field("value", self.SPACE.plus))
        verbalize = self.verbalizer_union(rules)
        self.verbalizer = verbalize + (self.INSERT_SPACE + verbalize).star
