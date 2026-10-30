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
from pynini.lib.pynutil import delete, insert

from tn.processor import Processor
from tn.utils import get_abs_path


class Cardinal(Processor):

    def __init__(self):
        super().__init__("cardinal", token_orders={})
        self.build_graphs()
        self.build_tagger()
        self.build_verbalizer()

    def build_graphs(self):
        digit = pynini.string_file(get_abs_path("korean/data/number/digit.tsv"))
        zero = pynini.string_file(get_abs_path("korean/data/number/zero.tsv"))
        tens = pynini.string_file(get_abs_path("korean/data/number/tens.tsv"))
        other_digit = (self.DIGIT - pynini.union("0", "1")) @ digit
        small = digit | tens + (digit | delete("0"))
        hundred = (pynini.cross("1", "백") | other_digit + insert("백")) + (
            delete("0").star | delete("0").star + small)
        thousand = (pynini.cross("1", "천") | other_digit + insert("천")) + (
            delete("0").star | hundred | delete("0").star + small)
        small = (small | self.DIGIT**3 @ hundred | self.DIGIT**4 @ thousand).optimize()
        padded = (self.DIGIT**4 @ (delete("0").star + small)).optimize()

        # Each four-digit block is consumed once; zero blocks omit their scale.
        # Unlike arbitrary whole-value swaps, this is a finite positional graph.
        scales = ["", "만", "억", "조", "경"]
        forms = [small, zero]
        lower = pynini.accep("")
        for index, scale in enumerate(scales):
            if index:
                top = small + insert(scale)
                if scale == "만":
                    top = (pynini.difference(self.input_projection(small), "1") @ small) + insert(scale)
                    top |= pynini.cross("1", "만")
                if index == 4:
                    top = self.DIGIT @ top
                forms.append(top + lower)
            block = padded + insert(scale) | delete("0000")
            lower = block + lower
        self.graph = pynini.union(*forms).optimize()
        self.digits = (digit | zero).optimize()
        native_ones = pynini.string_file(get_abs_path("korean/data/number/native_ones.tsv"))
        native_tens = pynini.string_file(get_abs_path("korean/data/ordinal/tens.tsv"))
        native_prefix = pynini.string_file(get_abs_path("korean/data/ordinal/tens_prefix.tsv"))
        self.native = (native_ones | native_tens | native_prefix + native_ones).optimize()
        self.counters = pynini.string_file(get_abs_path("korean/data/number/counter_suffix.tsv"))
        self.sign = pynini.string_map([("-", "마이너스 "), ("+", ""), ("(", "마이너스 ")])
        self.sign_tagger = self.tag_field("negative", self.sign) + insert(" ")
        self.sign_verbalizer = self.verbalize_field("negative", self.sign) + delete(" ")
        self.signals = pynini.union("번호는", "번호가", "번호를", "연락처는", "연락처가", "연락처를")
        separator = delete(pynini.union("-", ".", " ")).ques
        self.serial = ((self.digits + separator)**2 + self.digits + (separator + self.digits).star).optimize()

    def build_tagger(self):
        integer = self.tag_field("integer", self.graph)
        signed = self.sign_tagger.ques + integer
        parens = self.tag_field("negative", pynini.accep("(")) + insert(" ") + delete(" ").star
        parens += integer + delete(" ").star + delete(")")
        counter = self.tag_field("integer", self.native) + insert(" ") + self.tag_field("counter", self.counters)
        serial = self.tag_field("signal", self.signals) + self.DELETE_SPACE + insert(" ")
        serial += self.tag_field("serial", self.serial)
        self.tagger = self.add_tokens(signed | parens | counter | serial)

    def build_verbalizer(self):
        integer = self.verbalize_field("integer", self.graph)
        counter = self.verbalize_field("integer", self.native) + self.verbalize_field(
            "counter", self.counters, leading_space=True)
        serial = self.verbalize_field("signal", self.signals) + insert(" ")
        serial += self.verbalize_field("serial", self.serial, leading_space=True)
        self.verbalizer = self.delete_tokens(self.sign_verbalizer.ques + (integer | counter) | serial)
