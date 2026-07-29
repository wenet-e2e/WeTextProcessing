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

from pynini import string_file
from pynini.lib.pynutil import add_weight, delete, insert

from itn.chinese.rules.cardinal import Cardinal
from tn.processor import Processor
from itn.utils import get_abs_path


class Money(Processor):

    def __init__(self, enable_0_to_9=True, cardinal=None):
        super().__init__(name="money")
        self.enable_0_to_9 = enable_0_to_9
        self.cardinal = cardinal or Cardinal()
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        code = string_file(get_abs_path("chinese/data/money/code.tsv"))
        symbol = string_file(get_abs_path("chinese/data/money/symbol.tsv"))
        digit = string_file(get_abs_path("chinese/data/number/digit.tsv"))  # 1 ~ 9

        number = self.cardinal.number if self.enable_0_to_9 else self.cardinal.number_exclude_0_to_9
        # 七八美元 => $7~8
        number |= digit + insert("~") + digit
        # 三千三百八十元五毛八分 => ¥3380.58
        self.value = number
        self.currency = symbol | add_weight(code, 1)
        self.decimal = (insert(".") + digit + (delete("毛") | delete("角")) + (digit + delete("分")).ques).ques
        tagger = (self.tag_field("value", self.value) + insert(" ") + self.tag_field("currency", self.currency) + insert(" ") +
                  self.tag_field("decimal", self.decimal))
        self.tagger = self.add_tokens(tagger)

    def build_verbalizer(self):
        currency = self.verbalize_field("currency", self.currency)
        value = self.verbalize_field("value", self.value, leading_space=True)
        decimal = self.verbalize_field("decimal", self.decimal, leading_space=True)
        verbalizer = currency + value + decimal
        self.verbalizer = self.delete_tokens(verbalizer)
