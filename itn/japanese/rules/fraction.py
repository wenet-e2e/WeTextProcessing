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
from tn.utils import get_abs_path


class Fraction(Processor):

    def __init__(self):
        super().__init__(name="fraction")
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        cardinal = Cardinal(enable_million=True).number
        decimal = Cardinal(enable_million=True).decimal
        sign = string_file(get_abs_path("../itn/japanese/data/number/sign.tsv"))
        sign = insert('sign: "') + sign + insert('"')

        fraction_word = delete("分の") | delete(" 分 の　") | delete("分 の　") | delete("分 の")
        root_word = accep("√") | cross("ルート", "√")

        # denominator
        denominator = (decimal | (cardinal + root_word + cardinal) | (root_word + cardinal) | cardinal) + delete(
            " "
        ).ques
        denominator = insert('denominator: "') + denominator + insert('"')

        # numerator
        numerator = closure(delete(" ")) + (decimal | cardinal + root_word + cardinal | root_word + cardinal | cardinal)
        numerator = insert('numerator: "') + numerator + insert('"')

        # fraction
        fraction_sign = sign + insert(" ") + denominator + insert(" ") + fraction_word + numerator
        fraction_no_sign = denominator + insert(" ") + fraction_word + numerator
        regular_fractions = fraction_sign | fraction_no_sign

        integer_fraction_sign = (sign + insert(" ")).ques + denominator + insert(" ") + fraction_word + numerator
        fraction = regular_fractions | integer_fraction_sign
        self.tagger = self.add_tokens(fraction).optimize()

    def build_verbalizer(self):
        sign = delete('sign: "') + self.SIGMA + delete('"')
        denominator = delete('denominator: "') + self.SIGMA + delete('"')
        numerator = delete('numerator: "') + self.SIGMA + delete('"')
        fraction = (sign + delete(" ")).ques + numerator + delete(" ") + insert("/") + denominator
        self.verbalizer = self.delete_tokens(fraction).optimize()
