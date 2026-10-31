# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from tn.processor import Processor, RuleSpec
from tn.vietnamese.rules import date, decimal, fraction, time
from tn.vietnamese.rules.cardinal import Cardinal
from tn.vietnamese.rules.date import Date
from tn.vietnamese.rules.decimal import Decimal
from tn.vietnamese.rules.fraction import Fraction
from tn.vietnamese.rules.measure import Measure
from tn.vietnamese.rules.money import Money
from tn.vietnamese.rules.ordinal import Ordinal
from tn.vietnamese.rules.punctuation import Punctuation
from tn.vietnamese.rules.range import Range
from tn.vietnamese.rules.roman import Roman
from tn.vietnamese.rules.time import Time
from tn.vietnamese.rules.whitelist import WhiteList
from tn.vietnamese.rules.word import Word


TOKEN_ORDERS = {}
for _module in (date, decimal, fraction, time):
    TOKEN_ORDERS.update(_module.TOKEN_ORDER)


class Normalizer(Processor):

    def __init__(self, cache_dir=None, overwrite_cache=False):
        super().__init__(name="vi_normalizer", token_orders=TOKEN_ORDERS)
        self.build_fst("vi_tn", cache_dir, overwrite_cache, {})

    def build_tagger_and_verbalizer(self):
        cardinal = Cardinal()
        decimal = Decimal(cardinal)
        fraction = Fraction(cardinal)
        date = Date(cardinal)
        time = Time(cardinal)
        money = Money(cardinal, decimal)
        measure = Measure(cardinal, decimal, fraction)
        punctuation = Punctuation()
        word = Word()
        rules = (
            RuleSpec(WhiteList(), 1.01),
            RuleSpec(money, 1.1),
            RuleSpec(Range((time, date, decimal, money, measure)), 1.1),
            RuleSpec(decimal, 1.1),
            RuleSpec(date, 1.1),
            RuleSpec(cardinal, 1.1),
            RuleSpec(Ordinal(cardinal), 1.1),
            RuleSpec(fraction, 1.1),
            RuleSpec(time, 1.1),
            RuleSpec(measure, 1.1),
            RuleSpec(word, 100, verbalize=False),
            RuleSpec(Roman(cardinal), 101),
        )
        classify = self.tagger_union(rules) | word.fallback_tagger
        punct = add_weight(punctuation.tagger, 2.1)
        token = punct.star + classify + punct.star
        self.tagger = self.DELETE_SPACE + token + (self.DELETE_EXTRA_SPACE.ques + token).star + self.DELETE_SPACE
        self.tagger |= word.add_tokens(word.tag_field("value", self.SPACE.plus))
        self.verbalizer = self.join_tokens(self.verbalizer_union(rules), (word.verbalizer, punctuation.verbalizer))

    def join_tokens(self, semantic, plain):
        """Joins verbalized tokens with spaces, except before closing and after opening marks.

        This is NeMo's Vietnamese punctuation post-processing, expressed over
        token boundaries instead of a rewrite of the whole output string.
        """

        closing = pynini.union(*map(pynini.escape, Punctuation.CLOSING))
        opening = pynini.union(*map(pynini.escape, Punctuation.OPENING))
        starts = (closing + self.VSIGMA).optimize()
        ends = (self.VSIGMA + opening).optimize()
        neither = pynini.difference(self.VSIGMA, starts | ends).optimize()
        both = pynini.intersect(starts, ends).optimize()
        plain = pynini.union(*plain).optimize()

        def outputs(acceptor):
            return (plain @ acceptor).optimize()

        nn = (semantic | outputs(neither)).optimize()
        sn = outputs(pynini.difference(starts, ends))
        ne = outputs(pynini.difference(ends, starts))
        se = outputs(both)
        space = self.INSERT_SPACE
        # After an opening mark (or at the start) no space is inserted; otherwise
        # a space separates tokens unless the next one starts with a closing mark.
        glued_n, glued_e = nn | sn, ne | se
        spaced_n, spaced_e = space + nn | sn, space + ne | se
        after_space = (spaced_n | spaced_e + glued_e.star + glued_n).star + (spaced_e + glued_e.star).ques
        return (glued_e.star + (glued_n + after_space).ques).optimize()
