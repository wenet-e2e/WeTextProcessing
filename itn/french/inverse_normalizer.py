# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from itn.french.rules import cardinal, date, decimal, electronic, fraction, measure, money, ordinal, time
from itn.french.rules.cardinal import Cardinal
from itn.french.rules.date import Date
from itn.french.rules.decimal import Decimal
from itn.french.rules.electronic import Electronic
from itn.french.rules.fraction import Fraction
from itn.french.rules.measure import Measure
from itn.french.rules.money import Money
from itn.french.rules.ordinal import Ordinal
from itn.french.rules.punctuation import Punctuation
from itn.french.rules.telephone import Telephone
from itn.french.rules.time import Time
from itn.french.rules.whitelist import WhiteList
from itn.french.rules.word import Word
from tn.processor import Processor, RuleSpec


TOKEN_ORDERS = {}
for _module in (cardinal, date, decimal, electronic, fraction, measure, money, ordinal, time):
    TOKEN_ORDERS.update(_module.TOKEN_ORDER)


class InverseNormalizer(Processor):

    def __init__(self, cache_dir=None, overwrite_cache=False):
        super().__init__(name="fr_inverse_normalizer", ordertype="itn", token_orders=TOKEN_ORDERS)
        self.build_fst("fr_itn", cache_dir, overwrite_cache, {})

    def build_tagger_and_verbalizer(self):
        cardinal = Cardinal()
        decimal = Decimal(cardinal)
        fraction = Fraction(cardinal)
        punctuation = Punctuation()
        word = Word()
        rules = (
            RuleSpec(WhiteList(), 1.01),
            RuleSpec(Time(), 1.05),
            RuleSpec(Date(cardinal), 1.09),
            RuleSpec(decimal, 1.08),
            RuleSpec(Measure(cardinal, decimal, fraction), 1.1),
            RuleSpec(cardinal, 1.1),
            RuleSpec(Ordinal(cardinal), 1.1),
            RuleSpec(fraction, 1.09),
            RuleSpec(Money(cardinal, decimal), 1.07),
            RuleSpec(Telephone(), 1.1),
            RuleSpec(Electronic(), 1.1),
            RuleSpec(word, 100),
            RuleSpec(punctuation),
        )
        # Punctuation attaches to a neighbouring token ("vingt!" -> "20 !"); other tokens need whitespace.
        classify = self.tagger_union(rules)
        punct = add_weight(punctuation.tagger, 1.1)
        token = punct.star + classify + punct.star
        self.tagger = self.DELETE_SPACE + token + (self.DELETE_EXTRA_SPACE + token).star + self.DELETE_SPACE
        self.tagger |= word.add_tokens(word.tag_field("value", self.SPACE.plus))
        verbalize = self.verbalizer_union(rules)
        self.verbalizer = verbalize + (self.INSERT_SPACE + verbalize).star
