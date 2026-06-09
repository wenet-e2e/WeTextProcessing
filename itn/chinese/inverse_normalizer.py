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

from importlib_resources import files
from pynini.lib.pynutil import add_weight, delete

from itn.chinese.rules.cardinal import Cardinal
from itn.chinese.rules.char import Char
from itn.chinese.rules.date import Date
from itn.chinese.rules.train_number import TrainNumber
from itn.chinese.rules.fraction import Fraction
from itn.chinese.rules.license_plate import LicensePlate
from itn.chinese.rules.math import Math
from itn.chinese.rules.measure import Measure
from itn.chinese.rules.money import Money
from itn.chinese.rules.postprocessor import PostProcessor
from itn.chinese.rules.time import Time
from itn.chinese.rules.whitelist import Whitelist
from tn.processor import Processor


class InverseNormalizer(Processor):

    def __init__(
        self,
        cache_dir=None,
        overwrite_cache=False,
        remove_interjections=True,
        enable_standalone_number=True,
        enable_0_to_9=False,
        enable_million=False,
    ):
        super().__init__(name="zh_inverse_normalizer", ordertype="itn")
        self.remove_interjections = remove_interjections
        self.convert_number = enable_standalone_number
        self.enable_0_to_9 = enable_0_to_9
        self.enable_million = enable_million
        if cache_dir is None:
            cache_dir = files("itn")
        self.build_fst("zh_itn", cache_dir, overwrite_cache)

    def build_tagger_and_verbalizer(self):
        cardinal = Cardinal(self.convert_number, self.enable_0_to_9, self.enable_million)
        char = Char()
        date = Date()
        fraction = Fraction(cardinal=cardinal)
        train_number = TrainNumber()
        math = Math(cardinal=cardinal)
        measure = Measure(enable_0_to_9=self.enable_0_to_9, cardinal=cardinal)
        money = Money(enable_0_to_9=self.enable_0_to_9, cardinal=cardinal)
        time = Time()
        license_plate = LicensePlate()
        whitelist = Whitelist()

        tagger = (
            add_weight(date.tagger, 1.02)
            | add_weight(whitelist.tagger, 1.01)
            | add_weight(fraction.tagger, 1.05)
            | add_weight(measure.tagger, 1.05)
            | add_weight(money.tagger, 1.04)
            | add_weight(time.tagger, 1.05)
            | add_weight(cardinal.tagger, 1.06)
            | add_weight(math.tagger, 1.10)
            | add_weight(license_plate.tagger, 1.0)
            | add_weight(train_number.tagger, 1.0)
            | add_weight(char.tagger, 100)
        ).optimize()

        tagger = tagger.star
        self.tagger = tagger @ self.build_rule(delete(" "), "", "[EOS]")

        verbalizer = (
            cardinal.verbalizer
            | char.verbalizer
            | date.verbalizer
            | fraction.verbalizer
            | train_number.verbalizer
            | math.verbalizer
            | measure.verbalizer
            | money.verbalizer
            | time.verbalizer
            | license_plate.verbalizer
            | whitelist.verbalizer
        ).optimize()
        postprocessor = PostProcessor(remove_interjections=self.remove_interjections).processor

        self.verbalizer = (verbalizer @ postprocessor).star
