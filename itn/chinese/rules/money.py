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

from itn.chinese.rules.cardinal import Cardinal
from tn.processor import Processor
from tn.utils import get_abs_path

from pynini import string_file
from pynini.lib.pynutil import delete, insert


class Money(Processor):

    def __init__(self, enable_0_to_9=True):
        super().__init__(name='money')
        self.enable_0_to_9 = enable_0_to_9
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        code = string_file(get_abs_path('../itn/chinese/data/money/code.tsv'))
        symbol = string_file(
            get_abs_path('../itn/chinese/data/money/symbol.tsv'))
        digit = string_file(
            get_abs_path('../itn/chinese/data/number/digit.tsv'))  # 1 ~ 9

        number = Cardinal().number if self.enable_0_to_9 else \
            Cardinal().number_exclude_0_to_9
        # 七八美元 => $7~8
        number |= digit + insert("~") + digit
        # 三千三百八十元五毛八分 => ¥3380.58
        tagger = (insert('value: "') + number + insert('"') +
                  insert(' currency: "') + (code | symbol) + insert('"') +
                  insert(' decimal: "') +
                  (insert(".") + digit + (delete("毛") | delete("角")) +
                   (digit + delete("分")).ques).ques + insert('"'))
        self.tagger = self.add_tokens(tagger)

    def build_verbalizer(self):
        currency = delete('currency: "') + self.SIGMA + delete('"')
        value = delete(' value: "') + self.SIGMA + delete('"')
        decimal = delete(' decimal: "') + self.SIGMA + delete('"')
        verbalizer = currency + value + decimal
        self.verbalizer = self.delete_tokens(verbalizer)
