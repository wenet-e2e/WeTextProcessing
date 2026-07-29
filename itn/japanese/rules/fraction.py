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

from pynini import accep, closure, cross, string_file
from pynini.lib.pynutil import delete, insert

from itn.japanese.rules.cardinal import Cardinal
from tn.processor import Processor
from itn.utils import get_abs_path


class Fraction(Processor):

    def __init__(self, cardinal=None, input_processor=None):
        super().__init__(name="fraction")
        self.cardinal = cardinal or Cardinal(enable_million=True)
        self.input_processor = input_processor
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        cardinal = self.cardinal.number
        decimal = self.cardinal.decimal
        self.sign = self.apply_input_processor(
            string_file(get_abs_path("japanese/data/number/sign.tsv")),
            self.input_processor,
        )

        fraction_word = delete("分の") | delete(" 分 の　") | delete("分 の　") | delete("分 の")
        root_word = accep("√") | cross("ルート", "√")

        # denominator
        self.denominator = self.apply_input_processor(
            (decimal | (cardinal + root_word + cardinal) | (root_word + cardinal) | cardinal) + delete(" ").ques +
            fraction_word,
            self.input_processor,
        )

        # numerator
        self.numerator = self.apply_input_processor(
            closure(delete(" ")) + (decimal | cardinal + root_word + cardinal | root_word + cardinal | cardinal),
            self.input_processor,
        )

        # fraction
        fraction = ((self.tag_field("sign", self.sign) + insert(" ")).ques + self.tag_field("denominator", self.denominator) +
                    insert(" ") + self.tag_field("numerator", self.numerator))
        self.tagger = self.add_tokens(fraction).optimize()

    def build_verbalizer(self):
        sign = self.verbalize_field("sign", self.sign)
        denominator = self.verbalize_field("denominator", self.denominator)
        numerator = self.verbalize_field("numerator", self.numerator)
        fraction = (sign + delete(" ")).ques + numerator + delete(" ") + insert("/") + denominator
        self.verbalizer = self.delete_tokens(fraction).optimize()
