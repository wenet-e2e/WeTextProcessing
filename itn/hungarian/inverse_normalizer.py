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
from pynini.lib import pynutil

from itn.hungarian.rules.cardinal import Cardinal
from itn.hungarian.rules.date import Date
from itn.hungarian.rules.decimal import Decimal
from itn.hungarian.rules.electronic import Electronic
from itn.hungarian.rules.fraction import Fraction
from itn.hungarian.rules.measure import Measure
from itn.hungarian.rules.money import Money
from itn.hungarian.rules.ordinal import Ordinal
from itn.hungarian.rules.punctuation import Punctuation
from itn.hungarian.rules.telephone import Telephone
from itn.hungarian.rules.time import Time
from itn.hungarian.rules.whitelist import WhiteList
from itn.hungarian.rules.word import Word
from tn.processor import Processor, RuleSpec

# Every Hungarian rule emits its fields in written order.
TOKEN_ORDERS = {}

# Written without a space before (closing) or after (opening) the mark.
CLOSING_MARKS = ".,!?:;)]}…»”"
OPENING_MARKS = "([{«„“"


class InverseNormalizer(Processor):

    def __init__(self, cache_dir=None, overwrite_cache=False):
        super().__init__(name="hu_inverse_normalizer", ordertype="itn", token_orders=TOKEN_ORDERS)
        self.build_fst("hu_itn", cache_dir, overwrite_cache, {})

    def build_tagger_and_verbalizer(self):
        cardinal = Cardinal()
        ordinal = Ordinal(cardinal)
        decimal = Decimal(cardinal)
        fraction = Fraction(cardinal, ordinal)
        punctuation = Punctuation()
        word = Word(punctuation)
        rules = (
            RuleSpec(WhiteList(), 1.01),
            RuleSpec(Time(cardinal), 1.05),
            RuleSpec(Money(cardinal, decimal), 1.07),
            RuleSpec(Measure(cardinal, decimal, fraction), 1.08),
            RuleSpec(decimal, 1.08),
            RuleSpec(fraction, 1.09),
            RuleSpec(Date(cardinal), 1.09),
            RuleSpec(ordinal, 1.1),
            RuleSpec(cardinal, 1.1),
            RuleSpec(Telephone(cardinal), 1.1),
            RuleSpec(Electronic(), 1.1),
            RuleSpec(word, 200),
            RuleSpec(punctuation),
        )
        classify = self.tagger_union(rules)
        space = self.DELETE_EXTRA_SPACE
        punct = pynutil.add_weight(punctuation.tagger, 2.1)
        punct = (punct + (space.ques + punct).star).optimize()
        separator = space | space.ques + punct + space.ques
        tokens = (punct + space.ques).ques + classify + (separator + classify).star + (space.ques + punct).ques
        self.tagger = self.DELETE_SPACE + (tokens | punct) + self.DELETE_SPACE
        self.tagger |= word.add_tokens(word.tag_field("value", self.SPACE.plus))

        # Closing marks attach to the previous token, opening marks to the next.
        def marks(chars, graph=None):
            chars = pynini.union(*[pynini.escape(mark) for mark in chars]).plus
            value = chars if graph is None else pynini.difference(graph, chars)
            return punctuation.delete_tokens(punctuation.verbalize_field("value", value))

        closing = marks(CLOSING_MARKS)
        opening = marks(OPENING_MARKS)
        other = marks(CLOSING_MARKS, pynini.difference(punctuation.graph, pynini.union(
            *[pynini.escape(mark) for mark in OPENING_MARKS]).plus))
        plain = (self.verbalizer_union(rules[:-1]) | other).optimize()
        item = opening.star + plain | opening.plus + closing
        self.verbalizer = (
            (item | closing) + (self.INSERT_SPACE + item | closing).star + (self.INSERT_SPACE + opening.plus).ques
            | opening.plus
        )
