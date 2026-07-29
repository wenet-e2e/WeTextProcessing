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

from tn.japanese.rules.cardinal import Cardinal
from tn.japanese.rules.char import Char
from tn.japanese.rules.date import Date
from tn.japanese.rules.fraction import Fraction
from tn.japanese.rules.math import Math
from tn.japanese.rules.measure import Measure
from tn.japanese.rules.money import Money
from tn.japanese.rules.postprocessor import PostProcessor
from tn.japanese.rules.preprocessor import PreProcessor
from tn.japanese.rules.range import Range
from tn.japanese.rules.sport import Sport
from tn.japanese.rules.time import Time
from tn.japanese.rules.transliteration import Transliteration
from tn.japanese.rules.whitelist import Whitelist
from tn.processor import Processor, RuleSpec

TOKEN_ORDERS = {
    "date": ["year", "month", "day"],
    "fraction": ["denominator", "numerator"],
    "measure": ["denominator", "numerator", "value"],
    "money": ["value", "currency"],
    "time": ["noon", "hour", "minute", "second"],
}


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
        super().__init__(name="ja_normalizer", token_orders=TOKEN_ORDERS)
        self.transliterate = transliterate
        self.remove_interjections = remove_interjections
        self.remove_puncts = remove_puncts
        self.full_to_half = full_to_half
        self.tag_oov = tag_oov
        self.build_fst(
            "ja_tn",
            cache_dir,
            overwrite_cache,
            {
                "full_to_half": self.full_to_half,
                "remove_interjections": self.remove_interjections,
                "remove_puncts": self.remove_puncts,
                "tag_oov": self.tag_oov,
                "transliterate": self.transliterate,
            },
        )

    def build_tagger_and_verbalizer(self):
        input_normalizer = PreProcessor(full_to_half=self.full_to_half).processor
        cardinal = Cardinal(input_normalizer=input_normalizer)
        char = Char()
        range_rule = Range(input_normalizer=input_normalizer)
        date = Date(cardinal=cardinal, input_normalizer=input_normalizer, range_tagger=range_rule.tagger)
        fraction = Fraction(cardinal=cardinal, input_normalizer=input_normalizer)
        math = Math(cardinal=cardinal, input_normalizer=input_normalizer)
        measure = Measure(cardinal=cardinal, input_normalizer=input_normalizer)
        money = Money(cardinal=cardinal, input_normalizer=input_normalizer)
        sport = Sport(cardinal=cardinal, input_normalizer=input_normalizer)
        time = Time(input_normalizer=input_normalizer, range_tagger=range_rule.tagger)
        whitelist = Whitelist(input_normalizer=input_normalizer)

        rules = [
            RuleSpec(cardinal, 1.06),
            RuleSpec(char, 100),
            RuleSpec(date, 1.02),
            RuleSpec(fraction, 1.05),
            RuleSpec(math, 90),
            RuleSpec(measure, 1.05),
            RuleSpec(money, 1.05),
            RuleSpec(sport, 1.06),
            RuleSpec(time, 1.05),
            RuleSpec(whitelist, 1.03),
            RuleSpec(range_rule),
        ]
        if self.transliterate:
            transliteration = Transliteration(input_normalizer=input_normalizer)
            rules.append(RuleSpec(transliteration, 1.04))
        tagger = self.tagger_union(rules).star
        self.tagger = tagger @ self.build_rule(delete(" "), r="[EOS]")

        verbalizer = self.verbalizer_union(rules)

        postprocessor = PostProcessor(
            remove_interjections=self.remove_interjections,
            remove_puncts=self.remove_puncts,
            full_to_half=self.full_to_half,
            tag_oov=self.tag_oov,
        ).processor
        self.verbalizer = (verbalizer @ postprocessor).star
