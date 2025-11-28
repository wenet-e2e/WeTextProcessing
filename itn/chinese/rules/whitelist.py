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
from pynini.lib.pynutil import insert

from tn.processor import Processor
from tn.utils import get_abs_path


class Whitelist(Processor):

    def __init__(self):
        super().__init__(name="whitelist")
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        whitelist = string_file(get_abs_path("../itn/chinese/data/default/whitelist.tsv"))

        tagger = insert('value: "') + whitelist + insert('"')
        self.tagger = self.add_tokens(tagger)
