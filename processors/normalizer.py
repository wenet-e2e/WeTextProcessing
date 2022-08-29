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

import os

from processors.cardinal import Cardinal
from processors.char import Char
from processors.date import Date
from processors.fraction import Fraction
from processors.math import Math
from processors.measure import Measure
from processors.money import Money
from processors.postprocessor import PostProcessor
from processors.preprocessor import PreProcessor
from processors.processor import Processor
from processors.whitelist import Whitelist
from processors.time import Time

from pynini import Far
from pynini.lib.pynutil import add_weight, delete, insert


class Normalizer(Processor):

    def __init__(self, cache_dir=None, overwrite_cache=False):
        super().__init__(name='normalizer')
        self.cache_dir = cache_dir
        self.overwrite_cache = overwrite_cache

        far_file = None
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            far_file = os.path.join(self.cache_dir, 'zh_tn_normalizer.far')

        if far_file and os.path.exists(far_file):
            self.tagger = Far(far_file)['tagger']
            self.verbalizer = Far(far_file)['verbalizer']
        else:
            self.build_tagger()
            self.build_verbalizer()

        if self.cache_dir and self.overwrite_cache:
            self.export(far_file)

    def build_tagger(self):
        tagger = (add_weight(Date().tagger, 1.02)
                  | add_weight(Whitelist().tagger, 1.03)
                  | add_weight(Fraction().tagger, 1.05)
                  | add_weight(Measure().tagger, 1.05)
                  | add_weight(Money().tagger, 1.05)
                  | add_weight(Time().tagger, 1.05)
                  | add_weight(Cardinal().tagger, 1.06)
                  | add_weight(Math().tagger, 1.08)
                  | add_weight(Char().tagger, 100))
        # insert space between tokens, and remove the last space
        tagger = self.build_rule(tagger + insert(' '))
        tagger @= self.build_rule(delete(' '), '', '[EOS]')

        processor = PreProcessor(remove_interjections=True,
                                 full_to_half=True).processor
        self.tagger = processor @ tagger.optimize()

    def build_verbalizer(self):
        verbalizer = (Cardinal().verbalizer
                      | Char().verbalizer
                      | Date().verbalizer
                      | Fraction().verbalizer
                      | Math().verbalizer
                      | Measure().verbalizer
                      | Money().verbalizer
                      | Time().verbalizer
                      | Whitelist().verbalizer).optimize()
        verbalizer = (verbalizer + delete(' ').ques).star

        processor = PostProcessor(remove_puncts=False, tag_oov=True).processor
        self.verbalizer = verbalizer @ processor
