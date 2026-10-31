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

from pynini.lib import pynutil

from itn.german.rules import cardinal, date, decimal, electronic, fraction, measure, money
from itn.german.rules import ordinal, punctuation, telephone, time, whitelist, word
from itn.german.rules.cardinal import Cardinal
from itn.german.rules.date import Date
from itn.german.rules.decimal import Decimal
from itn.german.rules.electronic import Electronic
from itn.german.rules.fraction import Fraction
from itn.german.rules.measure import Measure
from itn.german.rules.money import Money
from itn.german.rules.ordinal import Ordinal
from itn.german.rules.punctuation import Punctuation
from itn.german.rules.telephone import Telephone
from itn.german.rules.time import Time
from itn.german.rules.whitelist import WhiteList
from itn.german.rules.word import Word
from tn.german.rules.cardinal import Cardinal as TNCardinal
from tn.german.rules.date import Date as TNDate
from tn.german.rules.decimal import Decimal as TNDecimal
from tn.german.rules.electronic import Electronic as TNElectronic
from tn.german.rules.fraction import Fraction as TNFraction
from tn.german.rules.ordinal import Ordinal as TNOrdinal
from tn.german.rules.time import Time as TNTime
from tn.german.rules.whitelist import WhiteList as TNWhiteList
from tn.processor import Processor, RuleSpec


TOKEN_ORDERS = {}
for _module in (cardinal, date, decimal, electronic, fraction, measure, money,
                ordinal, punctuation, telephone, time, whitelist, word):
    TOKEN_ORDERS.update(_module.TOKEN_ORDER)


class InverseNormalizer(Processor):
    def __init__(self, cache_dir=None, overwrite_cache=False):
        super().__init__(name="de_inverse_normalizer", ordertype="itn", token_orders=TOKEN_ORDERS)
        self.build_fst("de_itn", cache_dir, overwrite_cache, {})

    def build_tagger_and_verbalizer(self):
        tn_cardinal = TNCardinal(False)
        tn_ordinal = TNOrdinal(tn_cardinal, False)
        cardinal = Cardinal(tn_cardinal)
        decimal = Decimal(cardinal, TNDecimal(tn_cardinal, False))
        fraction = Fraction(cardinal, TNFraction(tn_cardinal, tn_ordinal, False))
        punctuation = Punctuation()
        word = Word(punctuation)
        rules = (
            RuleSpec(cardinal, 1.1),
            RuleSpec(WhiteList(TNWhiteList(False)), 1.0),
            RuleSpec(Time(TNTime(tn_cardinal, False)), 1.1),
            RuleSpec(Date(cardinal, TNDate(tn_cardinal, tn_ordinal, False), tn_ordinal), 1.1),
            RuleSpec(decimal, 1.1),
            RuleSpec(Measure(cardinal, decimal, fraction), 1.1),
            RuleSpec(Ordinal(cardinal, tn_ordinal), 1.1),
            RuleSpec(fraction, 1.1),
            RuleSpec(Money(cardinal, decimal), 1.1),
            RuleSpec(Telephone(tn_cardinal), 1.1),
            RuleSpec(Electronic(TNElectronic(False)), 1.1),
            RuleSpec(word, 100),
            RuleSpec(punctuation, 1.1),
        )
        classify = self.tagger_union(rules)
        punct = pynutil.add_weight(punctuation.tagger, 1.1)
        token = punct.star + classify + punct.star
        self.tagger = self.DELETE_SPACE + token + (self.DELETE_EXTRA_SPACE + token).star + self.DELETE_SPACE
        self.tagger |= punct.plus
        self.tagger |= word.add_tokens(word.tag_field("value", self.SPACE.plus))
        verbalize = self.verbalizer_union(rules)
        self.verbalizer = verbalize + (self.INSERT_SPACE + verbalize).star
