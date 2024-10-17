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

from tn.japanese.rules.cardinal import Cardinal
from tn.processor import Processor
from tn.utils import get_abs_path

from pynini import string_file
from pynini.lib.pynutil import delete, insert


class Math(Processor):

    def __init__(self):
        super().__init__(name="math")
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        operator = string_file(get_abs_path("japanese/data/math/operator.tsv"))

        number = Cardinal().number
        operator = number + (delete(" ").ques + operator + delete(" ").ques +
                             number).star
        bigger = operator + (delete(" ").ques + delete(">") + insert("は") +
                             delete(" ").ques + number + insert("より大きい")).star
        smaller = operator + (delete(" ").ques + delete("<") + insert("は") +
                              delete(" ").ques + number + insert("より小きい")).star
        no_less = operator + (delete(" ").ques +
                              (delete("≥") | delete(">=")) + insert("は") +
                              delete(" ").ques + number + insert("以上")).star
        no_more = operator + (delete(" ").ques +
                              (delete("≤") | delete("<=")) + insert("は") +
                              delete(" ").ques + number + insert("以下")).star
        tagger = (operator) | (bigger) | (smaller) | (no_less) | (no_more)
        tagger = insert('value: "') + tagger + insert('"')
        self.tagger = self.add_tokens(tagger)
