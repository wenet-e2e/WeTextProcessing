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

from pynini import cross, string_file
from pynini.lib.pynutil import delete, insert

from itn.japanese.rules.cardinal import Cardinal
from tn.processor import Processor
from tn.utils import get_abs_path


class Measure(Processor):

    def __init__(self, enable_0_to_9=True):
        super().__init__(name="measure")
        self.enable_0_to_9 = enable_0_to_9
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        unit_en = string_file(get_abs_path("../itn/japanese/data/measure/unit_en.tsv"))
        unit_ja = string_file(get_abs_path("../itn/japanese/data/measure/unit_ja.tsv"))

        cardinal = Cardinal().number if self.enable_0_to_9 else Cardinal().number_exclude_0_to_9
        decimal = Cardinal().decimal

        suffix = (
            insert("/")
            + (delete("每") | delete("毎"))
            + (unit_en | cross("時", "h") | cross("分", "min") | cross("秒", "s"))
        )

        measure = (cardinal | decimal) + unit_en + suffix.ques | (cardinal | decimal) + unit_ja

        tagger = insert('value: "') + measure + insert('"')
        self.tagger = self.add_tokens(tagger)
