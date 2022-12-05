# Copyright (c) 2022 Xingchen Song (sxc19@tsinghua.org.cn)
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

from tn.processor import Processor
from itn.chinese.rules.cardinal import Cardinal
from itn.chinese.rules.char import Char
from itn.chinese.rules.date import Date
from itn.chinese.rules.fraction import Fraction
from itn.chinese.rules.math import Math
from itn.chinese.rules.measure import Measure
from itn.chinese.rules.money import Money
from itn.chinese.rules.whitelist import Whitelist
from itn.chinese.rules.time import Time
from itn.chinese.rules.preprocessor import PreProcessor

from pynini import Far
from pynini.lib.pynutil import add_weight, delete
from importlib_resources import files


class InverseNormalizer(Processor):

    def __init__(self, cache_dir=None, overwrite_cache=False,
                 enable_standalone_number=True,
                 enable_0_to_9=True):
        super().__init__(name='inverse_normalizer', ordertype='itn')
        self.cache_dir = cache_dir
        self.overwrite_cache = overwrite_cache
        self.convert_number = enable_standalone_number
        self.enable_0_to_9 = enable_0_to_9

        far_file = files('itn').joinpath('zh_itn_normalizer.far')
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            far_file = os.path.join(self.cache_dir, 'zh_itn_normalizer.far')

        if far_file and os.path.exists(far_file) and not overwrite_cache:
            self.tagger = Far(far_file)['tagger']
            self.verbalizer = Far(far_file)['verbalizer']
        else:
            self.build_tagger()
            self.build_verbalizer()

        if self.cache_dir and self.overwrite_cache:
            self.export(far_file)

    def build_tagger(self):
        tagger = (add_weight(Date().tagger, 1.02)
                  | add_weight(Whitelist().tagger, 1.01)
                  | add_weight(Fraction().tagger, 1.05)
                  | add_weight(Measure().tagger, 1.05)
                  | add_weight(Money().tagger, 1.05)
                  | add_weight(Time().tagger, 1.05)
                  | add_weight(Cardinal(self.convert_number, self.enable_0_to_9).tagger, 1.06)
                  | add_weight(Math().tagger, 1.10)
                  | add_weight(Char().tagger, 100)).optimize()

        processor = PreProcessor(remove_interjections=True).processor
        tagger = (processor @ tagger).star
        # remove the last space
        self.tagger = tagger @ self.build_rule(delete(' '), '', '[EOS]')

    def build_verbalizer(self):
        verbalizer = (Cardinal(self.convert_number, self.enable_0_to_9).verbalizer
                      | Char().verbalizer
                      | Date().verbalizer
                      | Fraction().verbalizer
                      | Math().verbalizer
                      | Measure().verbalizer
                      | Money().verbalizer
                      | Time().verbalizer
                      | Whitelist().verbalizer).optimize().star

        self.verbalizer = verbalizer
