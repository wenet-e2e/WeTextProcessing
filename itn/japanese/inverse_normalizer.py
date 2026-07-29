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

from pynini.lib.pynutil import delete

from itn.japanese.rules.cardinal import Cardinal
from itn.japanese.rules.char import Char
from itn.japanese.rules.date import Date
from itn.japanese.rules.fraction import Fraction
from itn.japanese.rules.math import Math
from itn.japanese.rules.measure import Measure
from itn.japanese.rules.money import Money
from itn.japanese.rules.ordinal import Ordinal
from itn.japanese.rules.preprocessor import PreProcessor
from itn.japanese.rules.time import Time
from itn.japanese.rules.whitelist import Whitelist
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
        full_to_half=False,
        enable_standalone_number=True,
        enable_0_to_9=False,
        enable_million=False,
    ):
        super().__init__(name="ja_inverse_normalizer", ordertype="itn", token_orders=TOKEN_ORDERS)
        self.full_to_half = full_to_half
        self.convert_number = enable_standalone_number
        self.enable_0_to_9 = enable_0_to_9
        self.enable_million = enable_million
        self.build_fst(
            "ja_itn",
            cache_dir,
            overwrite_cache,
            {
                "enable_0_to_9": self.enable_0_to_9,
                "enable_million": self.enable_million,
                "enable_standalone_number": self.convert_number,
                "full_to_half": self.full_to_half,
            },
        )

    def build_tagger_and_verbalizer(self):
        processor = PreProcessor(full_to_half=self.full_to_half).processor
        cardinal = Cardinal(
            self.convert_number,
            self.enable_0_to_9,
            self.enable_million,
            input_processor=processor,
        )
        cardinal_million = Cardinal(enable_million=True, input_processor=processor)
        char = Char(input_processor=processor)
        date = Date(cardinal=cardinal, input_processor=processor)
        fraction = Fraction(cardinal=cardinal_million, input_processor=processor)
        math = Math(cardinal=cardinal, input_processor=processor)
        measure = Measure(
            enable_0_to_9=self.enable_0_to_9,
            cardinal=cardinal,
            input_processor=processor,
        )
        money = Money(
            enable_0_to_9=self.enable_0_to_9,
            cardinal=cardinal,
            input_processor=processor,
        )
        ordinal = Ordinal(cardinal=cardinal, input_processor=processor)
        time = Time(input_processor=processor)
        whitelist = Whitelist(input_processor=processor)

        rules = (
            RuleSpec(cardinal, 1.06),
            RuleSpec(char, 100),
            RuleSpec(date, 1.02),
            RuleSpec(fraction, 1.05),
            RuleSpec(math, 90),
            RuleSpec(measure, 1.05),
            RuleSpec(money, 1.04),
            RuleSpec(ordinal, 1.04),
            RuleSpec(time, 1.04),
            RuleSpec(whitelist, 1.01),
        )
        tagger = self.tagger_union(rules).star
        self.tagger = tagger @ self.build_rule(delete(" "), "", "[EOS]")

        self.verbalizer = self.verbalizer_union(rules).star
