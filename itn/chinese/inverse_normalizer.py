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

from pynini.lib.pynutil import delete

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
from tn.processor import Processor, RuleSpec

TOKEN_ORDERS = {
    "date": ["year", "month", "day", "preserve_order"],
    "fraction": ["sign", "numerator", "denominator"],
    "measure": ["numerator", "denominator", "value", "units"],
    "money": ["currency", "value", "decimal", "quantity"],
    "time": ["hour", "minute", "second", "noon", "zone"],
    "telephone": ["country_code", "number_part"],
    "electronic": ["username", "domain", "protocol"],
}


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
        super().__init__(name="zh_inverse_normalizer", ordertype="itn", token_orders=TOKEN_ORDERS)
        self.remove_interjections = remove_interjections
        self.convert_number = enable_standalone_number
        self.enable_0_to_9 = enable_0_to_9
        self.enable_million = enable_million
        self.build_fst(
            "zh_itn",
            cache_dir,
            overwrite_cache,
            {
                "enable_0_to_9": self.enable_0_to_9,
                "enable_million": self.enable_million,
                "enable_standalone_number": self.convert_number,
                "remove_interjections": self.remove_interjections,
            },
        )

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

        rules = (
            RuleSpec(date, 1.02),
            RuleSpec(whitelist, 1.01),
            RuleSpec(fraction, 1.05),
            RuleSpec(measure, 1.05),
            RuleSpec(money, 1.04),
            RuleSpec(time, 1.05),
            RuleSpec(cardinal, 1.06),
            RuleSpec(math, 1.10),
            RuleSpec(license_plate, 1.0),
            RuleSpec(train_number, 1.0),
            RuleSpec(char, 100),
        )
        tagger = self.tagger_union(rules)

        tagger = tagger.star
        self.tagger = tagger @ self.build_rule(delete(" "), "", "[EOS]")

        verbalizer = self.verbalizer_union(rules)
        postprocessor = PostProcessor(remove_interjections=self.remove_interjections).processor

        self.verbalizer = (verbalizer @ postprocessor).star
