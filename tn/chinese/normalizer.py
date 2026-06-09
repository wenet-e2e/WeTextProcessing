# Copyright (c) 2022 Zhendong Peng (pzd17@tsinghua.org.cn)
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

from tn.chinese.rules.cardinal import Cardinal
from tn.chinese.rules.char import Char
from tn.chinese.rules.date import Date
from tn.chinese.rules.fraction import Fraction
from tn.chinese.rules.math import Math
from tn.chinese.rules.measure import Measure
from tn.chinese.rules.money import Money
from tn.chinese.rules.postprocessor import PostProcessor
from tn.chinese.rules.preprocessor import PreProcessor
from tn.chinese.rules.sport import Sport
from tn.chinese.rules.time import Time
from tn.chinese.rules.whitelist import Whitelist
from tn.processor import Processor


class Normalizer(Processor):

    def __init__(
        self,
        cache_dir=None,
        overwrite_cache=False,
        remove_interjections=True,
        remove_erhua=True,
        traditional_to_simple=True,
        remove_puncts=False,
        full_to_half=True,
        tag_oov=False,
    ):
        super().__init__(name="zh_normalizer")
        self.remove_interjections = remove_interjections
        self.remove_erhua = remove_erhua
        self.traditional_to_simple = traditional_to_simple
        self.remove_puncts = remove_puncts
        self.full_to_half = full_to_half
        self.tag_oov = tag_oov
        if cache_dir is None:
            cache_dir = files("tn")
        self.build_fst("zh_tn", cache_dir, overwrite_cache)

    def build_tagger_and_verbalizer(self):
        processor = PreProcessor(traditional_to_simple=self.traditional_to_simple).processor
        cardinal = Cardinal()
        date = Date()
        whitelist = Whitelist(remove_erhua=self.remove_erhua)
        sport = Sport(cardinal=cardinal)
        fraction = Fraction(cardinal=cardinal)
        measure = Measure(cardinal=cardinal)
        money = Money(cardinal=cardinal)
        time = Time()
        math = Math(cardinal=cardinal)
        char = Char()

        tagger = (
            add_weight(date.tagger, 1.02)
            | add_weight(whitelist.tagger, 1.03)
            | add_weight(sport.tagger, 1.04)
            | add_weight(fraction.tagger, 1.05)
            | add_weight(measure.tagger, 1.05)
            | add_weight(money.tagger, 1.05)
            | add_weight(time.tagger, 1.05)
            | add_weight(cardinal.tagger, 1.06)
            | add_weight(math.tagger, 90)
            | add_weight(char.tagger, 100)
        ).optimize()
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

        postprocessor = PostProcessor(
            remove_interjections=self.remove_interjections,
            remove_puncts=self.remove_puncts,
            full_to_half=self.full_to_half,
            tag_oov=self.tag_oov,
        ).processor
        self.verbalizer = (verbalizer @ postprocessor).star
