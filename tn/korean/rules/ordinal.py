# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
from pynini.lib.pynutil import insert

from tn.processor import Processor
from tn.utils import get_abs_path


class Ordinal(Processor):
    """Ordinals with ``번째``, e.g. ``1번째`` -> ``첫번째``.

    Numbers below 40 use native Korean numerals, larger ones Sino-Korean
    numerals. A trailing particle such as ``가`` or ``까지`` is kept as is.
    """

    def __init__(self, cardinal):
        super().__init__("ordinal", token_orders={})
        self.cardinal = cardinal
        self.build_graphs()
        self.build_tagger()
        self.build_verbalizer()

    def build_graphs(self):
        digit = pynini.string_file(get_abs_path("korean/data/ordinal/digit.tsv"))
        zero = pynini.string_file(get_abs_path("korean/data/number/zero.tsv"))
        exceptions = pynini.string_file(get_abs_path("korean/data/ordinal/exceptions.tsv"))
        tens = pynini.string_file(get_abs_path("korean/data/ordinal/tens.tsv"))
        tens_prefix = pynini.string_file(get_abs_path("korean/data/ordinal/tens_prefix.tsv"))
        native = exceptions | digit | zero | tens | tens_prefix + digit
        # Own a single-digit alphabet: another rule's in-place closure must not
        # widen the 40+ filter to include native ordinals below forty.
        digit_inputs = pynini.union(*"0123456789")
        from_40 = pynini.union(*"456789") + digit_inputs | digit_inputs**3 + digit_inputs.star
        self.graph = (native | from_40 @ self.cardinal.graph).optimize()
        self.suffix = pynini.accep("번째")
        single = pynini.union("가", "이", "은", "는", "를", "을", "로", "도", "다")
        multi = pynini.union("부터", "까지")
        self.particle = (single | multi | single + multi).optimize()

    def build_tagger(self):
        tagger = self.tag_field("integer", self.graph) + insert(" ") + self.tag_field("suffix", self.suffix)
        tagger += (insert(" ") + self.tag_field("particle", self.particle)).ques
        self.tagger = self.add_tokens(tagger)

    def build_verbalizer(self):
        verbalizer = self.verbalize_field("integer", self.graph)
        verbalizer += self.verbalize_field("suffix", self.suffix, leading_space=True)
        verbalizer += self.verbalize_field("particle", self.particle, leading_space=True).ques
        self.verbalizer = self.delete_tokens(verbalizer)
