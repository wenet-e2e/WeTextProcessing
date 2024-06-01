# Copyright (c) 2022 Di Wu (1176705630@qq.com)
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

from tn.processor import Processor

from pynini import accep, string_file
from pynini.lib.pynutil import add_weight, delete, insert


class Whitelist(Processor):

    def __init__(self, asr=False, tts=False, symbol=False):
        super().__init__(name='whitelist')
        self.asr = asr
        self.tts = tts
        self.symbol = symbol
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        whitelist = string_file('english/data/whitelist/default.tsv')
        if self.asr:
            whitelist |= string_file('english/data/whitelist/asr.tsv')
        if self.tts:
            whitelist |= string_file('english/data/whitelist/tts.tsv')
        if self.symbol:
             whitelist |= string_file('english/data/whitelist/symbol.tsv')
        whitelist = whitelist.optimize()
        tagger = insert('value: "') + whitelist + insert('"')
        self.tagger = self.add_tokens(tagger)

    def build_verbalizer(self):
        super().build_verbalizer()
