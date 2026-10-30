# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from tn.processor import Processor, RuleSpec
from tn.russian.rules import time
from tn.russian.rules.cardinal import Cardinal
from tn.russian.rules.date import Date
from tn.russian.rules.decimal import Decimal
from tn.russian.rules.electronic import Electronic
from tn.russian.rules.measure import Measure
from tn.russian.rules.money import Money
from tn.russian.rules.ordinal import Ordinal
from tn.russian.rules.punctuation import Punctuation
from tn.russian.rules.telephone import Telephone
from tn.russian.rules.time import Time
from tn.russian.rules.whitelist import WhiteList
from tn.russian.rules.word import Word


TOKEN_ORDERS = {}
for _module in (time,):
    TOKEN_ORDERS.update(_module.TOKEN_ORDER)


class Normalizer(Processor):
    """Russian TN. Like NeMo, the grammar is non-deterministic: ``normalize``
    returns the best reading and ``normalize(text, nbest=n)`` the alternatives
    (cases, genders and spellings) that NeMo leaves to audio-based selection.
    """

    def __init__(self, cache_dir=None, overwrite_cache=False, deterministic=False):
        if deterministic:
            raise ValueError("Russian TN only supports non-deterministic normalization (deterministic=False).")
        super().__init__(name="ru_normalizer", token_orders=TOKEN_ORDERS)
        self.deterministic = deterministic
        self.build_fst("ru_tn", cache_dir, overwrite_cache, {"deterministic": deterministic})

    def build_tagger_and_verbalizer(self):
        cardinal = Cardinal()
        decimal = Decimal(cardinal)
        punctuation = Punctuation()
        word = Word(punctuation)
        rules = (
            RuleSpec(WhiteList(), 1.01),
            RuleSpec(Time(), 1.1),
            RuleSpec(Date(), 1.09),
            RuleSpec(decimal, 1.1),
            RuleSpec(Measure(cardinal, decimal), 0.9),
            RuleSpec(cardinal, 1.1),
            RuleSpec(Ordinal(), 1.1),
            RuleSpec(Money(cardinal, decimal), 1.1),
            RuleSpec(Telephone(), 1.1),
            RuleSpec(Electronic(), 1.1),
            RuleSpec(word, 100),
            RuleSpec(punctuation),
        )
        classify = self.tagger_union(rules)
        punct = pynutil.add_weight(punctuation.tagger, 1.1)
        token = punct.star + classify + punct.star
        separator = pynutil.add_weight(self.DELETE_EXTRA_SPACE, 1.1)
        self.tagger = self.DELETE_SPACE + token + (separator + token).star + self.DELETE_SPACE
        self.tagger |= punct.plus
        self.tagger |= word.add_tokens(word.tag_field("value", self.SPACE.plus))
        verbalize = self.verbalizer_union(rules)
        self.verbalizer = verbalize + (self.INSERT_SPACE + verbalize).star
