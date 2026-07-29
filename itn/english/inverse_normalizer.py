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

from pynini import closure
from pynini.lib.pynutil import add_weight, delete

from itn.english.rules.cardinal import Cardinal
from itn.english.rules.char import Char
from itn.english.rules.date import Date
from itn.english.rules.decimal import Decimal
from itn.english.rules.electronic import Electronic
from itn.english.rules.measure import Measure
from itn.english.rules.money import Money
from itn.english.rules.ordinal import Ordinal
from itn.english.rules.punctuation import Punctuation
from itn.english.rules.telephone import Telephone
from itn.english.rules.time import Time
from itn.english.rules.whitelist import Whitelist
from itn.english.rules.word import Word
from tn.processor import Processor, RuleSpec

TOKEN_ORDERS = {
    "date": ["year", "month", "day", "preserve_order"],
    "fraction": ["sign", "numerator", "denominator"],
    "measure": ["numerator", "denominator", "value", "units"],
    "money": ["currency", "value", "decimal", "quantity"],
    "time": ["hour", "minute", "second", "noon", "zone"],
    "telephone": ["country_code", "number_part"],
    "electronic": ["username", "domain", "protocol"],
}


class InverseNormalizer(Processor):

    def __init__(self, cache_dir=None, overwrite_cache=False):
        super().__init__(name="en_inverse_normalizer", ordertype="itn", token_orders=TOKEN_ORDERS)
        self.build_fst("en_itn", cache_dir, overwrite_cache, {})

    def build_tagger_and_verbalizer(self):
        cardinal = Cardinal()
        ordinal = Ordinal(cardinal=cardinal)
        decimal = Decimal(cardinal=cardinal)
        date = Date(cardinal=cardinal, ordinal=ordinal)
        time = Time(cardinal=cardinal)
        measure = Measure(cardinal=cardinal, decimal=decimal)
        money = Money(cardinal=cardinal, decimal=decimal)
        telephone = Telephone(cardinal=cardinal)
        electronic = Electronic()
        whitelist = Whitelist()
        word = Word()
        char = Char()
        punctuation = Punctuation()

        rules = (
            RuleSpec(date, 1.09),
            RuleSpec(time, 1.1),
            RuleSpec(measure, 1.1),
            RuleSpec(money, 1.08),
            RuleSpec(whitelist, 1.01),
            RuleSpec(telephone, 1.1),
            RuleSpec(electronic, 1.1),
            RuleSpec(ordinal, 1.09),
            RuleSpec(decimal, 1.1),
            RuleSpec(cardinal, 1.1),
            RuleSpec(word, 50),
            RuleSpec(char, 100),
        )
        classify = self.tagger_union(rules)

        punct = add_weight(punctuation.tagger, 1.1)
        token = closure(punct + delete(" ").ques) + classify + closure(delete(" ").ques + punct)
        graph = token + closure(self.DELETE_EXTRA_SPACE + token)
        self.tagger = delete(" ").star + graph + delete(" ").star

        verbalizer_rules = list(rules)
        verbalizer_rules.append(RuleSpec(punctuation))
        verbalizer = self.verbalizer_union(verbalizer_rules)

        self.verbalizer = (verbalizer + self.INSERT_SPACE).star @ self.build_rule(self.DELETE_EXTRA_SPACE) @ self.build_rule(
            delete(" "), r="[EOS]")
