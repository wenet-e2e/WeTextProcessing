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

from pynini.lib import pynutil

from itn.english import inverse_normalizer as english
from itn.spanish import inverse_normalizer as spanish
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
from itn.spanish.rules.word import Word
from itn.spanish_english.rules import english as en_rules
from itn.spanish_english.rules.prefixed import Prefixed
from itn.spanish_english.rules.time import Time
from itn.spanish_english.rules.whitelist import Whitelist
from tn.processor import Processor, RuleSpec

PREFIX = "en_"
TOKEN_ORDERS = dict(spanish.TOKEN_ORDERS)
TOKEN_ORDERS.update({PREFIX + name: order for name, order in english.TOKEN_ORDERS.items()})


class InverseNormalizer(Processor):

    def __init__(self, cache_dir=None, overwrite_cache=False, cased=False):
        """``cased=True`` also accepts capitalized spoken words ("Doce Dólares")."""
        super().__init__(name="es_en_inverse_normalizer", ordertype="itn", token_orders=TOKEN_ORDERS)
        self.cased = cased
        self.build_fst("es_en_itn", cache_dir, overwrite_cache, {"cased": cased})

    def build_tagger_and_verbalizer(self):
        cased = self.cased
        cardinal = Cardinal(cased)
        ordinal = Ordinal(cardinal, cased)
        decimal = Decimal(cardinal, cased)
        fraction = Fraction(cardinal, ordinal, cased)

        en_cardinal = en_rules.Cardinal(cased=cased)
        en_ordinal = en_rules.Ordinal(cardinal=en_cardinal, cased=cased)
        en_decimal = en_rules.Decimal(cardinal=en_cardinal, cased=cased)

        def en(rule):
            return Prefixed(rule, PREFIX)

        punctuation = Punctuation()
        word = Word()
        rules = (
            # "doctor" is on both lists; the code-mixed grammar writes "dr.".
            RuleSpec(Whitelist("whitelist", "es_whitelist.tsv"), 1.011),
            RuleSpec(Whitelist(PREFIX + "whitelist", "en_whitelist.tsv"), 1.01),
            RuleSpec(Time(cased), 1.08),
            RuleSpec(en(en_rules.Time(cardinal=en_cardinal, cased=cased)), 1.1),
            RuleSpec(Date(cardinal, cased), 1.09),
            RuleSpec(en(en_rules.Date(cardinal=en_cardinal, ordinal=en_ordinal, cased=cased)), 1.09),
            RuleSpec(decimal, 1.09),
            RuleSpec(en(en_decimal), 1.09),
            RuleSpec(fraction, 1.09),
            RuleSpec(Measure(cardinal, decimal, fraction, cased), 1.6),
            RuleSpec(en(en_rules.Measure(cardinal=en_cardinal, decimal=en_decimal, cased=cased)), 1.1),
            RuleSpec(cardinal, 1.6),
            RuleSpec(en(en_cardinal), 1.1),
            RuleSpec(ordinal, 1.6),
            RuleSpec(en(en_ordinal), 1.09),
            RuleSpec(Money(cardinal, decimal, cased), 1.6),
            RuleSpec(en(en_rules.Money(cardinal=en_cardinal, decimal=en_decimal, cased=cased)), 1.1),
            RuleSpec(Telephone(cased), 1.6),
            RuleSpec(en(en_rules.Telephone(cardinal=en_cardinal, cased=cased)), 1.1),
            RuleSpec(Electronic(cased), 2.96),
            RuleSpec(en(en_rules.Electronic(cased=cased)), 1.1),
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
