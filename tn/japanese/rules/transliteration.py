# Copyright (c) 2025 Zhendong Peng (pzd17@tsinghua.org.cn)
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
from pynini.lib.pynutil import delete

from tn.processor import Processor
from tn.utils import get_abs_path


class Transliteration(Processor):

    def __init__(self, input_normalizer=None):
        super().__init__(name="transliteration")
        self.input_normalizer = input_normalizer
        self.transliteration = None
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        self.transliteration = self.apply_input_processor(
            string_file(get_abs_path("japanese/data/pyopenjtalk/transliteration.tsv")),
            self.input_normalizer,
        )
        self.tagger = self.add_tokens(self.tag_field("value", self.transliteration))

    def build_verbalizer(self):
        verbalizer = delete('value: "') + self.transliteration + delete('"')
        self.verbalizer = self.delete_tokens(verbalizer)
