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

from itn.japanese.rules.cardinal import Cardinal
from itn.japanese.rules.char import Char
from itn.japanese.rules.date import Date
from itn.japanese.rules.fraction import Fraction
from itn.japanese.rules.math import Math
from itn.japanese.rules.measure import Measure
from itn.japanese.rules.money import Money
from itn.japanese.rules.ordinal import Ordinal
from itn.japanese.rules.postprocessor import PostProcessor
from itn.japanese.rules.preprocessor import PreProcessor
from itn.japanese.rules.time import Time
from itn.japanese.rules.whitelist import Whitelist
from tn.processor import Processor


class InverseNormalizer(Processor):

    def __init__(
        self,
        cache_dir=None,
        overwrite_cache=False,
        full_to_half=False,
        enable_standalone_number=True,
        enable_0_to_9=False,
        enable_million=False,
    ):
        super().__init__(name="ja_inverse_normalizer", ordertype="itn")
        self.full_to_half = full_to_half
        self.convert_number = enable_standalone_number
        self.enable_0_to_9 = enable_0_to_9
        self.enable_million = enable_million
        if cache_dir is None:
            cache_dir = files("itn")
        self.build_fst("ja_itn", cache_dir, overwrite_cache)

    def build_tagger_and_verbalizer(self):
        processor = PreProcessor(full_to_half=self.full_to_half).processor
        cardinal = Cardinal(self.convert_number, self.enable_0_to_9, self.enable_million)
        cardinal_million = Cardinal(enable_million=True)
        char = Char()
        date = Date(cardinal=cardinal)
        fraction = Fraction(cardinal=cardinal_million)
        math = Math(cardinal=cardinal)
        measure = Measure(enable_0_to_9=self.enable_0_to_9, cardinal=cardinal)
        money = Money(enable_0_to_9=self.enable_0_to_9, cardinal=cardinal)
        ordinal = Ordinal(cardinal=cardinal)
        time = Time()
        whitelist = Whitelist()

        tagger = (
            add_weight(cardinal.tagger, 1.06)
            | add_weight(char.tagger, 100)
            | add_weight(date.tagger, 1.02)
            | add_weight(fraction.tagger, 1.05)
            | add_weight(math.tagger, 90)
            | add_weight(measure.tagger, 1.05)
            | add_weight(money.tagger, 1.04)
            | add_weight(ordinal.tagger, 1.04)
            | add_weight(time.tagger, 1.04)
            | add_weight(whitelist.tagger, 1.01)
        ).optimize().star
        tagger = (processor @ tagger).star
        self.tagger = tagger @ self.build_rule(delete(" "), "", "[EOS]")

        verbalizer = (
            cardinal.verbalizer
            | char.verbalizer
            | date.verbalizer
            | fraction.verbalizer
            | math.verbalizer
            | measure.verbalizer
            | money.verbalizer
            | ordinal.verbalizer
            | time.verbalizer
            | whitelist.verbalizer
        )

        postprocessor = PostProcessor().processor
        self.verbalizer = (verbalizer @ postprocessor).star
