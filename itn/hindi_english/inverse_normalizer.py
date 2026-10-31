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

from pynini.lib.pynutil import add_weight, delete

from itn.english.inverse_normalizer import TOKEN_ORDERS as EN_TOKEN_ORDERS
from itn.hindi.inverse_normalizer import TOKEN_ORDERS as HI_TOKEN_ORDERS
from itn.hindi.rules.cardinal import Cardinal
from itn.hindi.rules.date import Date
from itn.hindi.rules.decimal import Decimal
from itn.hindi.rules.fraction import Fraction
from itn.hindi.rules.measure import Measure
from itn.hindi.rules.money import Money
from itn.hindi.rules.ordinal import Ordinal
from itn.hindi.rules.punctuation import Punctuation
from itn.hindi.rules.telephone import Telephone
from itn.hindi.rules.time import Time
from itn.hindi.rules.whitelist import Whitelist
from itn.hindi.rules.word import Word
from itn.hindi_english.rules import english
from tn.processor import Processor, RuleSpec

# Hindi and English tokens share names (cardinal, date, money, ...); a token is verbalized by whichever
# language's verbalizer accepts its raw field text. The Hindi rules emit fields in written order, so only
# the English canonical orders are needed (Hindi money keeps its own order for its single "value" field).
TOKEN_ORDERS = dict(HI_TOKEN_ORDERS)
TOKEN_ORDERS.update(EN_TOKEN_ORDERS)

# Classifier weight of a punctuation mark glued to a token.
PUNCT_WEIGHT = 1.1


class InverseNormalizer(Processor):

    def __init__(self, cache_dir=None, overwrite_cache=False, cased=False):
        """``cased=True`` also accepts English spoken words with a capitalized first letter ("July twenty")."""
        super().__init__(name="hi_en_inverse_normalizer", ordertype="itn", token_orders=TOKEN_ORDERS)
        self.cased = cased
        self.build_fst("hi_en_itn", cache_dir, overwrite_cache, {"cased": cased})

    def build_tagger_and_verbalizer(self):
        cased = self.cased
        cardinal = Cardinal()
        decimal = Decimal(cardinal)
        en_cardinal = english.Cardinal(cased=cased)
        en_ordinal = english.Ordinal(cardinal=en_cardinal, cased=cased)
        en_decimal = english.Decimal(cardinal=en_cardinal, cased=cased)
        punctuation = Punctuation()
        word = Word()
        rules = (
            RuleSpec(Whitelist(), 1.01),
            RuleSpec(english.Whitelist(cased=cased), 1.01),
            RuleSpec(Time(cardinal), 1.1),
            RuleSpec(english.Time(cardinal=en_cardinal, cased=cased), 1.1),
            RuleSpec(Date(cardinal), 1.1),
            RuleSpec(english.Date(cardinal=en_cardinal, ordinal=en_ordinal, cased=cased), 1.09),
            RuleSpec(decimal, 1.1),
            RuleSpec(en_decimal, 1.1),
            RuleSpec(Fraction(cardinal), 1.1),
            RuleSpec(Measure(cardinal, decimal), 1.1),
            RuleSpec(english.Measure(cardinal=en_cardinal, decimal=en_decimal, cased=cased), 1.1),
            RuleSpec(cardinal, 1.1),
            RuleSpec(en_cardinal, 1.1),
            RuleSpec(Ordinal(cardinal), 1.1),
            RuleSpec(en_ordinal, 1.09),
            RuleSpec(Money(cardinal, decimal), 1.1),
            RuleSpec(english.Money(cardinal=en_cardinal, decimal=en_decimal, cased=cased), 1.08),
            RuleSpec(Telephone(), 1.1),
            RuleSpec(english.Telephone(cardinal=en_cardinal, cased=cased), 1.1),
            RuleSpec(english.Electronic(cased=cased), 1.1),
            RuleSpec(word, 100),
            RuleSpec(punctuation),
        )
        classify = self.tagger_union(rules)
        punct = add_weight(punctuation.tagger, PUNCT_WEIGHT)
        token = punct.star + classify + punct.star
        tagger = token + (delete(self.SPACE.plus) + token).star
        self.tagger = self.DELETE_SPACE + tagger + self.DELETE_SPACE
        self.tagger |= word.add_tokens(word.tag_field("value", self.SPACE.plus))

        verbalize = self.verbalizer_union(rules)
        self.verbalizer = verbalize + (self.INSERT_SPACE + verbalize).star
