# Copyright (c) 2022 Zhendong Peng (pzd17@tsinghua.org.cn)
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

    def __init__(self, remove_erhua=True):
        super().__init__(name='whitelist')
        self.remove_erhua = remove_erhua
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        whitelist = (string_file('tn/chinese/data/default/whitelist.tsv')
                     | string_file('tn/chinese/data/erhua/whitelist.tsv'))

        erhua = add_weight(insert('erhua: "') + accep('儿'), 0.1)
        tagger = (erhua | (insert('value: "') + whitelist)) + insert('"')
        self.tagger = self.add_tokens(tagger)

    def build_verbalizer(self):
        super().build_verbalizer()
        if self.remove_erhua:
            verbalizer = self.delete_tokens(delete('erhua: "儿"'))
        else:
            verbalizer = self.delete_tokens(
                delete('erhua: \"') + accep('儿') + delete('\"'))
        self.verbalizer |= verbalizer
