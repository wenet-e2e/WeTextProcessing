# Copyright (c) 2024 Logan Liu (2319277867@qq.com)
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
from tn.japanese.rules.cardinal import Cardinal
from tn.japanese.rules.char import Char
from tn.japanese.rules.date import Date
from tn.japanese.rules.fraction import Fraction
from tn.japanese.rules.math import Math
from tn.japanese.rules.measure import Measure
from tn.japanese.rules.money import Money
from tn.japanese.rules.postprocessor import PostProcessor
from tn.japanese.rules.preprocessor import PreProcessor
from tn.japanese.rules.sport import Sport
from tn.japanese.rules.time import Time
from tn.japanese.rules.transliteration import Transliteration
from tn.japanese.rules.whitelist import Whitelist

from pynini.lib.pynutil import add_weight, delete
from importlib_resources import files


class Normalizer(Processor):

    def __init__(self,
                 cache_dir=None,
                 overwrite_cache=False,
                 transliterate=False,
                 remove_interjections=False,
                 remove_puncts=False,
                 full_to_half=True,
                 tag_oov=False):
        super().__init__(name='ja_normalizer')
        self.transliterate = transliterate
        self.remove_interjections = remove_interjections
        self.remove_puncts = remove_puncts
        self.full_to_half = full_to_half
        self.tag_oov = tag_oov
        if cache_dir is None:
            cache_dir = files("tn")
        self.build_fst('ja_tn', cache_dir, overwrite_cache)

    def build_tagger(self):
        processor = PreProcessor(full_to_half=self.full_to_half).processor
        cardinal = add_weight(Cardinal().tagger, 1.06)
        char = add_weight(Char().tagger, 100)
        date = add_weight(Date().tagger, 1.02)
        fraction = add_weight(Fraction().tagger, 1.05)
        math = add_weight(Math().tagger, 90)
        measure = add_weight(Measure().tagger, 1.05)
        money = add_weight(Money().tagger, 1.05)
        sport = add_weight(Sport().tagger, 1.06)
        time = add_weight(Time().tagger, 1.05)
        whitelist = add_weight(Whitelist().tagger, 1.03)
        tagger = (cardinal | char | date | fraction | math | measure | money
                  | sport | time | whitelist).optimize()
        if self.transliterate:
            transliteration = add_weight(Transliteration().tagger, 1.04)
            tagger = (tagger | transliteration).optimize()
        tagger = (processor @ tagger).star
        self.tagger = tagger @ self.build_rule(delete(' '), r='[EOS]')

    def build_verbalizer(self):
        cardinal = Cardinal().verbalizer
        char = Char().verbalizer
        date = Date().verbalizer
        fraction = Fraction().verbalizer
        math = Math().verbalizer
        measure = Measure().verbalizer
        money = Money().verbalizer
        sport = Sport().verbalizer
        time = Time().verbalizer
        transliteration = Transliteration().verbalizer
        whitelist = Whitelist().verbalizer
        verbalizer = (cardinal | char | date | fraction | math | measure
                      | money | sport | time | whitelist).optimize()
        if self.transliterate:
            verbalizer = (verbalizer | transliteration).optimize()

        processor = PostProcessor(
            remove_interjections=self.remove_interjections,
            remove_puncts=self.remove_puncts,
            tag_oov=self.tag_oov).processor
        self.verbalizer = (verbalizer @ processor).star
