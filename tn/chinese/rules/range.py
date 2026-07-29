# Copyright (c) 2026, WENET COMMUNITY.
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

from pynini import cross
from pynini.lib.pynutil import delete

from tn.processor import Processor


class Range(Processor):
    """Tags range separators and verbalizes their spoken form."""

    def __init__(self):
        super().__init__(name="range")
        self.range = (cross("-", "到") | cross("~", "到")).optimize()
        self.tagger = self.add_tokens(self.tag_field("value", self.range))
        verbalizer = delete('value: "') + self.range + delete('"')
        self.verbalizer = self.delete_tokens(verbalizer)
