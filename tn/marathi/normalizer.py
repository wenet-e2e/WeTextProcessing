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

from tn.marathi.rules.cardinal import Cardinal
from tn.marathi.rules.date import Date
from tn.marathi.rules.decimal import Decimal
from tn.marathi.rules.punctuation import Punctuation
from tn.marathi.rules.time import Time
from tn.marathi.rules.word import Word
from tn.processor import Processor, RuleSpec

# Every Marathi rule emits its fields in spoken order, so no token needs
# reordering before verbalization.
TOKEN_ORDERS = {}


class Normalizer(Processor):

    def __init__(self, cache_dir=None, overwrite_cache=False):
        super().__init__(name="mr_normalizer", token_orders=TOKEN_ORDERS)
        self.build_fst("mr_tn", cache_dir, overwrite_cache, {})

    def build_tagger_and_verbalizer(self):
        cardinal = Cardinal()
        punctuation = Punctuation()
        word = Word()
        rules = (
            RuleSpec(Time(), 1.05),
            RuleSpec(Date(cardinal), 1.09),
            RuleSpec(Decimal(cardinal), 1.08),
            RuleSpec(cardinal, 1.1),
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
