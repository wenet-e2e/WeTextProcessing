# Copyright (c) 2026 Zhendong Peng (pzd17@tsinghua.org.cn)
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

import pynini
from pynini.lib.pynutil import delete, insert

from tn.hebrew.rules import numbers
from tn.hebrew.rules.utils import PREFIX, first_field
from tn.processor import Processor


class Cardinal(Processor):
    """Written integers, e.g. ``כ-1,000`` -> ``כאלף``.

    The tagger keeps the digits (with their commas), a ``-`` sign and a
    one-letter prefix written before a hyphen; the verbalizer reads the
    number and glues the prefix to it.
    """

    def __init__(self):
        super().__init__(name="cardinal")
        self.number = (numbers.grouped_digits() @ numbers.cardinal_graph()).optimize()
        # The written side of ``number``, cheap to restrict with differences.
        self.digits = pynini.project(numbers.grouped_digits(), "input").optimize()
        self.prefix = self.tag_field("prefix", PREFIX) + delete("-") + insert(" ")
        self.negative = self.tag_field("negative", pynini.accep("-")) + insert(" ")
        self.tagger = self.add_tokens(self.prefix.ques + self.negative.ques + self.tag_field("integer", self.number))

        self.verbalize_prefix = first_field(self, "prefix", PREFIX)
        self.verbalize_negative = first_field(self, "negative", pynini.cross("-", "מינוס "))
        self.verbalizer = self.delete_tokens(
            self.verbalize_prefix.ques + self.verbalize_negative.ques + self.verbalize_field("integer", self.number))
