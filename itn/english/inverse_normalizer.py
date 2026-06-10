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

from importlib_resources import files
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
from tn.processor import Processor


class InverseNormalizer(Processor):

    def __init__(self, cache_dir=None, overwrite_cache=False):
        super().__init__(name="en_inverse_normalizer", ordertype="itn")
        if cache_dir is None:
            cache_dir = files("itn")
        self.build_fst("en_itn", cache_dir, overwrite_cache)

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

        classify = (
            add_weight(date.tagger, 1.09)
            | add_weight(time.tagger, 1.1)
            | add_weight(measure.tagger, 1.1)
            | add_weight(money.tagger, 1.08)
            | add_weight(whitelist.tagger, 1.01)
            | add_weight(telephone.tagger, 1.1)
            | add_weight(electronic.tagger, 1.1)
            | add_weight(ordinal.tagger, 1.09)
            | add_weight(decimal.tagger, 1.1)
            | add_weight(cardinal.tagger, 1.1)
            | add_weight(word.tagger, 50)
            | add_weight(char.tagger, 100)
        ).optimize()

        punct = add_weight(punctuation.tagger, 1.1)
        token = closure(punct + delete(" ").ques) + classify + closure(delete(" ").ques + punct)
        graph = token + closure(self.DELETE_EXTRA_SPACE + token)
        self.tagger = delete(" ").star + graph + delete(" ").star

        verbalizer = (
            cardinal.verbalizer
            | ordinal.verbalizer
            | decimal.verbalizer
            | date.verbalizer
            | time.verbalizer
            | measure.verbalizer
            | money.verbalizer
            | telephone.verbalizer
            | electronic.verbalizer
            | whitelist.verbalizer
            | word.verbalizer
            | char.verbalizer
            | punctuation.verbalizer
        ).optimize()

        self.verbalizer = (verbalizer + self.INSERT_SPACE).star @ self.build_rule(
            self.DELETE_EXTRA_SPACE
        ) @ self.build_rule(delete(" "), r="[EOS]")
