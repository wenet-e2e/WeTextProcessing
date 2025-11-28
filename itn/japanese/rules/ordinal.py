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

from pynini import accep
from pynini.lib.pynutil import insert

from itn.japanese.rules.cardinal import Cardinal
from tn.processor import Processor


class Ordinal(Processor):

    def __init__(self):
        super().__init__(name="ordinal")
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        cardinal = Cardinal().number
        ordinal = (cardinal + accep("番目")) | (accep("第") + cardinal)
        tagger = insert('value: "') + ordinal + insert('"')
        self.tagger = self.add_tokens(tagger)
