# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from pynini import accep
from pynini.lib import pynutil

from tn.french.rules import cardinal, date, decimal, fraction, ordinal
from tn.french.rules.cardinal import Cardinal
from tn.french.rules.date import Date
from tn.french.rules.decimal import Decimal
from tn.french.rules.fraction import Fraction
from tn.french.rules.ordinal import Ordinal
from tn.french.rules.punctuation import Punctuation
from tn.french.rules.whitelist import WhiteList
from tn.french.rules.word import Word
from tn.processor import Processor, RuleSpec


TOKEN_ORDERS = {}
for _module in (cardinal, date, decimal, fraction, ordinal):
    TOKEN_ORDERS.update(_module.TOKEN_ORDER)


class Normalizer(Processor):

    def __init__(self, cache_dir=None, overwrite_cache=False):
        super().__init__(name="fr_normalizer", token_orders=TOKEN_ORDERS)
        self.build_fst("fr_tn", cache_dir, overwrite_cache, {})

    def build_tagger_and_verbalizer(self):
        cardinal = Cardinal()
        ordinal = Ordinal(cardinal)
        word = Word()
        punctuation = Punctuation()
        whitelist = WhiteList()
        rules = (
            RuleSpec(whitelist, 1.01),
            RuleSpec(Date(cardinal), 1.1),
            RuleSpec(cardinal, 1.1),
            RuleSpec(Fraction(cardinal, ordinal), 1.09),
            RuleSpec(ordinal, 1.1),
            RuleSpec(Decimal(cardinal), 1.1),
            RuleSpec(word, 200, verbalize=False),
            RuleSpec(punctuation, verbalize=False),
        )
        classify = self.tagger_union(rules)
        punct = pynutil.add_weight(punctuation.tagger, 2.1)
        percent = pynutil.add_weight(accep("%") @ whitelist.tagger, 1.01)
        separator = self.DELETE_EXTRA_SPACE | punct.plus | percent
        tokens = punct.star + classify + (separator + classify).star + percent.ques + punct.star
        self.tagger = self.DELETE_SPACE + tokens + self.DELETE_SPACE
        self.tagger |= punct.plus
        self.tagger |= word.add_tokens(word.tag_field("value", self.SPACE.plus))
        self.verbalizer = self.join_tokens(self.verbalizer_union(rules), (word.verbalizer, punctuation.verbalizer))
