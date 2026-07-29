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
from pynini.lib.pynutil import delete, insert

from tn.japanese.rules.cardinal import Cardinal
from tn.processor import Processor
from tn.utils import get_abs_path


class Money(Processor):

    def __init__(self, cardinal=None, input_normalizer=None):
        super().__init__(name="money")
        self.input_normalizer = input_normalizer
        self.cardinal = cardinal or Cardinal(input_normalizer=input_normalizer)
        self.currency = None
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        code = string_file(get_abs_path("japanese/data/money/code.tsv"))
        symbol = string_file(get_abs_path("japanese/data/money/symbol.tsv"))

        self.currency = self.apply_input_processor(code | symbol, self.input_normalizer)
        tagger = (self.tag_field("currency", self.currency) + delete(" ").ques + insert(" ") +
                  self.tag_field("value", self.cardinal.number))
        self.tagger = self.add_tokens(tagger)

    def build_verbalizer(self):
        value = delete('value: "') + self.cardinal.number + delete('" ')
        currency = delete('currency: "') + self.currency + delete('"')
        verbalizer = value + currency
        self.verbalizer = self.delete_tokens(verbalizer)
