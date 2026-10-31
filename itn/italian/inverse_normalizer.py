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

import pynini
from pynini.lib.pynutil import add_weight

from itn.italian.rules import money
from itn.italian.rules.cardinal import Cardinal
from itn.italian.rules.decimal import Decimal
from itn.italian.rules.electronic import Electronic
from itn.italian.rules.measure import Measure
from itn.italian.rules.money import Money
from itn.italian.rules.punctuation import Punctuation
from itn.italian.rules.time import Time
from itn.italian.rules.whitelist import WhiteList
from itn.italian.rules.word import Word
from tn.processor import Processor, RuleSpec

TOKEN_ORDERS = dict(money.TOKEN_ORDER)

# Written punctuation that attaches to the previous / next token.
CLOSING = ".,;:!?)}"
OPENING = "({"


class InverseNormalizer(Processor):

    def __init__(self, cache_dir=None, overwrite_cache=False):
        super().__init__(name="it_inverse_normalizer", ordertype="itn", token_orders=TOKEN_ORDERS)
        self.build_fst("it_itn", cache_dir, overwrite_cache, {})

    def build_tagger_and_verbalizer(self):
        cardinal = Cardinal()
        decimal = Decimal(cardinal)
        punctuation = Punctuation()
        word = Word()
        rules = (
            RuleSpec(WhiteList(), 1.01),
            RuleSpec(Time(), 1.05),
            RuleSpec(Money(cardinal, decimal), 1.07),
            RuleSpec(decimal, 1.08),
            RuleSpec(Measure(cardinal, decimal), 1.08),
            RuleSpec(cardinal, 1.1),
            RuleSpec(Electronic(), 1.1),
            RuleSpec(word, 100),
            RuleSpec(punctuation, 1.1, verbalize=False),
        )
        classify = self.tagger_union(rules)
        # Punctuation splits off only when attached to another token.
        punct = add_weight(punctuation.tagger, 1.1)
        token = punct.star + classify + punct.star
        self.tagger = self.DELETE_SPACE + token + (self.DELETE_EXTRA_SPACE + token).star + self.DELETE_SPACE
        self.tagger |= word.add_tokens(word.tag_field("value", self.SPACE.plus))

        def marks(chars):
            graph = pynini.union(*[pynini.escape(char) for char in chars])
            return punctuation.delete_tokens(punctuation.verbalize_field("value", graph))

        other_marks = [mark for mark in "!#$%&'()*+,-./:;<=>?@^_`{|}~" if mark not in CLOSING + OPENING]
        closing = marks(CLOSING)
        opening = marks(OPENING)
        other = self.verbalizer_union(rules) | marks(other_marks)
        # Written Italian attaches closing marks to the previous token and
        # opening marks to the next one: ``( cinque ) .`` -> ``(5).``.
        item = opening.star + other | add_weight(opening.plus, 1.0)
        self.verbalizer = (item | closing) + (closing | self.INSERT_SPACE + item).star
