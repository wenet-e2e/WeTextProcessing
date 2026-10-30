# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pynini
from pynini.lib import pynutil

from tn.processor import Processor
from tn.utils import get_abs_path


TOKEN_ORDER = {}


class Telephone(Processor):
    def __init__(self, cardinal, deterministic=True):
        super().__init__("telephone", token_orders=TOKEN_ORDER)
        self.deterministic = deterministic
        zero = pynini.string_file(get_abs_path("german/data/numbers/zero.tsv")).invert()
        digits = pynini.string_file(get_abs_path("german/data/numbers/digit.tsv")).invert()
        digits = digits | pynini.cross("1", "eins") | zero
        singles = (digits + pynutil.insert(" ")).star + digits
        two_digits = self.DIGIT ** 2 @ cardinal.two_digits
        country = pynini.cross("+", "plus ").ques + (two_digits | zero)
        country = country | (pynutil.delete("(") + zero + pynutil.insert(" ") + singles + pynutil.delete(")"))
        country = country | (zero + pynutil.insert(" ") + singles)
        self.country_code_graph = country.optimize()
        separator = pynini.cross(pynini.union("-", " "), " ")
        numbers = singles + (separator + singles).ques
        length = pynini.closure(self.DIGIT | pynini.union("-", " ", "(", ")"), 7)
        self.number_part_graph = (length @ numbers).optimize()
        self.fields_tagger = (
            self.tag_field("country_code", self.country_code_graph) + pynini.accep(" ")
            + self.tag_field("number_part", self.number_part_graph)
        ).optimize()
        self.fields_verbalizer = (
            self.verbalize_field("country_code", self.country_code_graph) + pynini.accep(" ")
            + self.verbalize_field("number_part", self.number_part_graph)
        ).optimize()
        self.tagger = self.add_tokens(self.fields_tagger)
        self.verbalizer = self.delete_tokens(self.fields_verbalizer)
