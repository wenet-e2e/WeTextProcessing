# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

from itn.armenian.rules.cardinal import Cardinal
from itn.armenian.rules.decimal import Decimal
from itn.armenian.rules.fraction import Fraction
from itn.armenian.rules.measure import Measure
from itn.armenian.rules.money import Money
from itn.armenian.rules.ordinal import Ordinal
from itn.armenian.rules.punctuation import Punctuation
from itn.armenian.rules.time import Time
from itn.armenian.rules.whitelist import WhiteList
from itn.armenian.rules.word import Word
from tn.processor import Processor, RuleSpec

# Every Armenian rule emits its fields in written order, so no token needs
# reordering before verbalization.
TOKEN_ORDERS = {}


class InverseNormalizer(Processor):

    def __init__(self, cache_dir=None, overwrite_cache=False):
        super().__init__(name="hy_inverse_normalizer", ordertype="itn", token_orders=TOKEN_ORDERS)
        self.build_fst("hy_itn", cache_dir, overwrite_cache, {})

    def build_tagger_and_verbalizer(self):
        cardinal = Cardinal()
        ordinal = Ordinal(cardinal)
        decimal = Decimal(cardinal)
        punctuation = Punctuation()
        word = Word()
        rules = (
            RuleSpec(WhiteList(), 1.01),
            RuleSpec(Time(), 1.05),
            RuleSpec(decimal, 1.08),
            RuleSpec(Measure(cardinal, decimal), 1.1),
            RuleSpec(cardinal, 1.1),
            RuleSpec(ordinal, 1.1),
            RuleSpec(Fraction(cardinal, ordinal), 1.09),
            RuleSpec(Money(cardinal, decimal), 1.07),
            RuleSpec(word, 100),
            RuleSpec(punctuation),
        )
        classify = self.tagger_union(rules)
        # Punctuation splits off only when attached to another token.
        punct = add_weight(punctuation.tagger, 1.1)
        token = punct.star + classify + punct.star
        self.tagger = self.DELETE_SPACE + token + (self.DELETE_EXTRA_SPACE + token).star + self.DELETE_SPACE
        self.tagger |= word.add_tokens(word.tag_field("value", self.SPACE.plus))
        verbalize = self.verbalizer_union(rules)
        self.verbalizer = verbalize + (self.INSERT_SPACE + verbalize).star
