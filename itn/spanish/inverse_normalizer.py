# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from itn.spanish.rules import cardinal, decimal, electronic, fraction, measure, money, time
from itn.spanish.rules.cardinal import Cardinal
from itn.spanish.rules.date import Date
from itn.spanish.rules.decimal import Decimal
from itn.spanish.rules.electronic import Electronic
from itn.spanish.rules.fraction import Fraction
from itn.spanish.rules.measure import Measure
from itn.spanish.rules.money import Money
from itn.spanish.rules.ordinal import Ordinal
from itn.spanish.rules.punctuation import Punctuation
from itn.spanish.rules.telephone import Telephone
from itn.spanish.rules.time import Time
from itn.spanish.rules.whitelist import Whitelist
from itn.spanish.rules.word import Word
from tn.processor import Processor, RuleSpec

TOKEN_ORDERS = {}
for _module in (cardinal, decimal, electronic, fraction, measure, money, time):
    TOKEN_ORDERS.update(_module.TOKEN_ORDER)


class InverseNormalizer(Processor):

    def __init__(self, cache_dir=None, overwrite_cache=False, cased=False):
        """``cased=True`` also accepts capitalized spoken words ("Doce Dólares")."""
        super().__init__(name="es_inverse_normalizer", ordertype="itn", token_orders=TOKEN_ORDERS)
        self.cased = cased
        self.build_fst("es_itn", cache_dir, overwrite_cache, {"cased": cased})

    def build_tagger_and_verbalizer(self):
        cased = self.cased
        cardinal = Cardinal(cased)
        ordinal = Ordinal(cardinal, cased)
        decimal = Decimal(cardinal, cased)
        fraction = Fraction(cardinal, ordinal, cased)
        punctuation = Punctuation()
        word = Word()
        rules = (
            RuleSpec(Whitelist(cased), 1.01),
            RuleSpec(Time(cased), 1.08),
            RuleSpec(Date(cardinal, cased), 1.09),
            RuleSpec(decimal, 1.09),
            RuleSpec(fraction, 1.09),
            RuleSpec(Measure(cardinal, decimal, fraction, cased), 1.6),
            RuleSpec(cardinal, 1.6),
            RuleSpec(ordinal, 1.6),
            RuleSpec(Money(cardinal, decimal, cased), 1.6),
            RuleSpec(Telephone(cased), 1.6),
            RuleSpec(Electronic(cased), 2.96),
            RuleSpec(word, 100),
            RuleSpec(punctuation),
        )
        classify = self.tagger_union(rules)
        # Punctuation may stick to either side of a token ("veinte!").
        punct = pynutil.add_weight(punctuation.tagger, 1.1).star
        token = punct + classify + punct
        self.tagger = self.DELETE_SPACE + token + (pynutil.delete(self.SPACE.plus) + token).star + self.DELETE_SPACE
        self.tagger |= word.add_tokens(word.tag_field("value", self.SPACE.plus))
        verbalize = self.verbalizer_union(rules)
        self.verbalizer = verbalize + (self.INSERT_SPACE + verbalize).star
