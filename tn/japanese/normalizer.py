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

from importlib_resources import files
from pynini.lib.pynutil import add_weight, delete

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
from tn.processor import Processor


class Normalizer(Processor):

    def __init__(
        self,
        cache_dir=None,
        overwrite_cache=False,
        transliterate=False,
        remove_interjections=False,
        remove_puncts=False,
        full_to_half=True,
        tag_oov=False,
    ):
        super().__init__(name="ja_normalizer")
        self.transliterate = transliterate
        self.remove_interjections = remove_interjections
        self.remove_puncts = remove_puncts
        self.full_to_half = full_to_half
        self.tag_oov = tag_oov
        if cache_dir is None:
            cache_dir = files("tn")
        self.build_fst("ja_tn", cache_dir, overwrite_cache)

    def build_tagger_and_verbalizer(self):
        processor = PreProcessor(full_to_half=self.full_to_half).processor
        cardinal = Cardinal()
        char = Char()
        date = Date(cardinal=cardinal)
        fraction = Fraction(cardinal=cardinal)
        math = Math(cardinal=cardinal)
        measure = Measure(cardinal=cardinal)
        money = Money(cardinal=cardinal)
        sport = Sport(cardinal=cardinal)
        time = Time()
        whitelist = Whitelist()

        tagger = (
            add_weight(cardinal.tagger, 1.06)
            | add_weight(char.tagger, 100)
            | add_weight(date.tagger, 1.02)
            | add_weight(fraction.tagger, 1.05)
            | add_weight(math.tagger, 90)
            | add_weight(measure.tagger, 1.05)
            | add_weight(money.tagger, 1.05)
            | add_weight(sport.tagger, 1.06)
            | add_weight(time.tagger, 1.05)
            | add_weight(whitelist.tagger, 1.03)
        ).optimize()
        if self.transliterate:
            transliteration = Transliteration()
            tagger = (tagger | add_weight(transliteration.tagger, 1.04)).optimize()
        tagger = (processor @ tagger).star
        self.tagger = tagger @ self.build_rule(delete(" "), r="[EOS]")

        verbalizer = (
            cardinal.verbalizer
            | char.verbalizer
            | date.verbalizer
            | fraction.verbalizer
            | math.verbalizer
            | measure.verbalizer
            | money.verbalizer
            | sport.verbalizer
            | time.verbalizer
            | whitelist.verbalizer
        ).optimize()
        if self.transliterate:
            verbalizer = (verbalizer | transliteration.verbalizer).optimize()

        postprocessor = PostProcessor(
            remove_interjections=self.remove_interjections, remove_puncts=self.remove_puncts, tag_oov=self.tag_oov
        ).processor
        self.verbalizer = (verbalizer @ postprocessor).star
