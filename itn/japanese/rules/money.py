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

from pynini import string_file
from pynini.lib.pynutil import insert

from itn.japanese.rules.cardinal import Cardinal
from tn.processor import Processor
from itn.utils import get_abs_path


class Money(Processor):

    def __init__(self, enable_0_to_9=True, cardinal=None, input_processor=None):
        super().__init__(name="money")
        self.enable_0_to_9 = enable_0_to_9
        self.cardinal = cardinal or Cardinal()
        self.input_processor = input_processor
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        symbol = string_file(get_abs_path("japanese/data/money/symbol.tsv"))

        number = self.cardinal.number if self.enable_0_to_9 else self.cardinal.number_exclude_0_to_9
        decimal = self.cardinal.decimal
        # 三千三百八十点五八円 => ¥3380.58
        self.value = self.apply_input_processor(number | decimal, self.input_processor)
        self.currency = self.apply_input_processor(symbol, self.input_processor)
        tagger = (self.tag_field("value", self.value) + insert(" ") + self.tag_field("currency", self.currency))
        self.tagger = self.add_tokens(tagger)

    def build_verbalizer(self):
        currency = self.verbalize_field("currency", self.currency)
        value = self.verbalize_field("value", self.value, leading_space=True)
        verbalizer = currency + value
        self.verbalizer = self.delete_tokens(verbalizer)
