# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from pynini.lib.pynutil import delete

from itn.korean.rules.cardinal import Cardinal
from itn.korean.rules.date import Date
from itn.korean.rules.decimal import Decimal
from itn.korean.rules.fraction import Fraction
from itn.korean.rules.measure import Measure
from itn.korean.rules.money import Money
from itn.korean.rules.ordinal import Ordinal
from itn.korean.rules.telephone import Telephone
from itn.korean.rules.time import Time
from itn.korean.rules.whitelist import Whitelist
from itn.korean.rules.word import Word
from tn.processor import Processor, RuleSpec


TOKEN_ORDERS = {
    "cardinal": ["negative", "integer"],
    "date": ["year", "month", "day"],
    "decimal": ["negative", "integer_part", "fractional_part", "quantity"],
    "fraction": ["negative", "integer_part", "numerator", "denominator", "suffix"],
    "measure": ["negative", "numerator", "denominator", "value", "units"],
    "money": ["currency", "value"],
    "ordinal": ["integer", "counter", "suffix"],
    "telephone": ["number_part"],
    "time": ["prefix", "hours", "minutes", "seconds", "suffix"],
    "whitelist": ["value"],
    "word": ["value"],
}


class InverseNormalizer(Processor):
    def __init__(self, cache_dir=None, overwrite_cache=False):
        super().__init__(name="ko_inverse_normalizer", ordertype="itn", token_orders=TOKEN_ORDERS)
        self.build_fst("ko_itn", cache_dir, overwrite_cache, {})

    def build_tagger_and_verbalizer(self):
        cardinal = Cardinal()
        decimal = Decimal(cardinal)
        word = Word()
        rules = (
            RuleSpec(cardinal, 1.1),
            RuleSpec(Ordinal(cardinal), 1.1),
            RuleSpec(decimal, 1.1),
            RuleSpec(Fraction(cardinal, decimal), 1.0),
            RuleSpec(Time(), 1.0),
            RuleSpec(Date(cardinal), 1.1),
            RuleSpec(Money(decimal), 1.1),
            RuleSpec(Telephone(), 1.1),
            RuleSpec(Measure(cardinal), 1.1),
            RuleSpec(Whitelist(), 1.01),
            RuleSpec(word, 100),
        )
        classify = self.tagger_union(rules)
        # A boundary is optional: adjacent zero words must remain separate tokens.
        self.tagger = self.DELETE_SPACE + classify + (self.DELETE_SPACE + classify).star + self.DELETE_SPACE
        self.tagger |= word.add_tokens(word.tag_field("value", self.SPACE.plus))
        verbalize = self.verbalizer_union(rules)
        self.verbalizer = verbalize + (self.INSERT_SPACE + verbalize).star
        self.verbalizer @= self.build_rule(self.DELETE_EXTRA_SPACE)
        self.verbalizer @= self.build_rule(delete(" "), r="[EOS]")
