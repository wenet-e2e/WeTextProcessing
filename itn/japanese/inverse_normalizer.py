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
from itn.japanese.rules.cardinal import Cardinal
from itn.japanese.rules.char import Char
from itn.japanese.rules.date import Date
from itn.japanese.rules.fraction import Fraction
from itn.japanese.rules.math import Math
from itn.japanese.rules.measure import Measure
from itn.japanese.rules.ordinal import Ordinal
from itn.japanese.rules.preprocessor import PreProcessor
from itn.japanese.rules.postprocessor import PostProcessor
from itn.japanese.rules.whitelist import Whitelist
from itn.japanese.rules.time import Time

from pynini.lib.pynutil import add_weight, delete
from importlib_resources import files


class InverseNormalizer(Processor):

    def __init__(self,
                 cache_dir=None,
                 overwrite_cache=False,
                 full_to_half=False,
                 enable_standalone_number=True,
                 enable_0_to_9=False,
                 enable_million=False):
        super().__init__(name='ja_inverse_normalizer', ordertype='itn')
        self.full_to_half = full_to_half
        self.convert_number = enable_standalone_number
        self.enable_0_to_9 = enable_0_to_9
        self.enable_million = enable_million
        if cache_dir is None:
            cache_dir = files("itn")
        self.build_fst('ja_itn', cache_dir, overwrite_cache)

    def build_tagger(self):
        processor = PreProcessor(full_to_half=self.full_to_half).processor

        cardinal = add_weight(
            Cardinal(self.convert_number, self.enable_0_to_9,
                     self.enable_million).tagger, 1.06)
        char = add_weight(Char().tagger, 100)
        date = add_weight(Date().tagger, 1.02)
        fraction = add_weight(Fraction().tagger, 1.05)
        math = add_weight(Math().tagger, 90)
        measure = add_weight(
            Measure(enable_0_to_9=self.enable_0_to_9).tagger, 1.05)
        ordinal = add_weight(Ordinal().tagger, 1.04)
        time = add_weight(Time().tagger, 1.04)
        whitelist = add_weight(Whitelist().tagger, 1.01)

        tagger = (cardinal | char | date | fraction | math | measure | ordinal
                  | time | whitelist).optimize().star
        tagger = (processor @ tagger).star
        # remove the last space
        self.tagger = tagger @ self.build_rule(delete(' '), '', '[EOS]')

    def build_verbalizer(self):
        cardinal = Cardinal(self.convert_number, self.enable_0_to_9).verbalizer
        char = Char().verbalizer
        date = Date().verbalizer
        fraction = Fraction().verbalizer
        math = Math().verbalizer
        measure = Measure().verbalizer
        ordinal = Ordinal().verbalizer
        time = Time().verbalizer
        whitelist = Whitelist().verbalizer

        verbalizer = cardinal | char | date | fraction | math | measure | ordinal | time | whitelist

        processor = PostProcessor().processor
        self.verbalizer = (verbalizer @ processor).star
