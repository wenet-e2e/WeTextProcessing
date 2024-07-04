# Copyright (c) 2022 Zhendong Peng (pzd17@tsinghua.org.cn)
# Copyright (c) 2024 Xingchen Song (sxc19@tsinghua.org.cn)
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

from tn.processor import Processor
from tn.english.rules.cardinal import Cardinal
from tn.english.rules.ordinal import Ordinal
from tn.english.rules.decimal import Decimal
from tn.english.rules.fraction import Fraction
from tn.english.rules.word import Word
from tn.english.rules.date import Date
from tn.english.rules.time import Time
from tn.english.rules.measure import Measure
from tn.english.rules.money import Money
from tn.english.rules.telephone import Telephone
from tn.english.rules.electronic import Electronic
from tn.english.rules.whitelist import WhiteList
from tn.english.rules.punctuation import Punctuation
from tn.english.rules.range import Range

from pynini.lib.pynutil import add_weight, delete
from importlib_resources import files


class Normalizer(Processor):

    def __init__(self, cache_dir=None, overwrite_cache=False):
        super().__init__(name='en_normalizer', ordertype="en_tn")
        if cache_dir is None:
            cache_dir = files("tn")
        self.build_fst('en_tn', cache_dir, overwrite_cache)

    def build_tagger(self):
        cardinal = add_weight(Cardinal().tagger, 1.0)
        ordinal = add_weight(Ordinal().tagger, 1.0)
        decimal = add_weight(Decimal().tagger, 1.0)
        fraction = add_weight(Fraction().tagger, 1.0)
        date = add_weight(Date().tagger, 0.99)
        time = add_weight(Time().tagger, 1.00)
        measure = add_weight(Measure().tagger, 1.00)
        money = add_weight(Money().tagger, 1.00)
        telephone = add_weight(Telephone().tagger, 1.00)
        electronic = add_weight(Electronic().tagger, 1.00)
        word = add_weight(Word().tagger, 100)
        whitelist = add_weight(WhiteList().tagger, 1.00)
        punct = add_weight(Punctuation().tagger, 2.00)
        rang = add_weight(Range().tagger, 1.01)
        # TODO(xcsong): add roman
        tagger = \
            (cardinal
             | ordinal
             | word
             | date
             | decimal
             | fraction
             | time
             | measure
             | money
             | telephone
             | electronic
             | whitelist
             | rang
             | punct
             ).optimize() + (punct.plus | self.DELETE_SPACE)
        # delete the first and last space
        self.tagger = (delete(' ').star + tagger.star) @ self.build_rule(
            delete(' '), r='[EOS]')

    def build_verbalizer(self):
        cardinal = Cardinal().verbalizer
        ordinal = Ordinal().verbalizer
        decimal = Decimal().verbalizer
        fraction = Fraction().verbalizer
        word = Word().verbalizer
        date = Date().verbalizer
        time = Time().verbalizer
        measure = Measure().verbalizer
        money = Money().verbalizer
        telephone = Telephone().verbalizer
        electronic = Electronic().verbalizer
        whitelist = WhiteList().verbalizer
        punct = Punctuation().verbalizer
        rang = Range().verbalizer
        verbalizer = \
            (cardinal
             | ordinal
             | word
             | date
             | decimal
             | fraction
             | time
             | measure
             | money
             | telephone
             | electronic
             | whitelist
             | punct
             | rang
             ).optimize() + (punct.plus | self.INSERT_SPACE)
        self.verbalizer = verbalizer.star @ self.build_rule(delete(' '),
                                                            r='[EOS]')
