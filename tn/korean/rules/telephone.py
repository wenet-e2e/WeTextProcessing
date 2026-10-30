# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

from tn.processor import Processor
from tn.utils import get_abs_path


class Telephone(Processor):

    def __init__(self):
        super().__init__("telephone", token_orders={})
        digit = (pynini.string_file(get_abs_path("korean/data/number/digit.tsv"))
                 | pynini.cross("0", "영")).optimize()
        separator = delete(pynini.union("-", ".", " "))
        first_block = digit**2 | digit**3
        first = (first_block + separator | delete("(") + first_block + delete(")") + separator.ques)
        self.number_part = (first + insert(" ") + (digit**3 | digit**4)
                            + separator + insert(" ") + digit**4).optimize()
        self.country_code = (delete("+") + insert("국가번호 ") + pynini.closure(digit, 1, 3)).optimize()

        country = self.tag_field("country_code", self.country_code)
        country += delete(pynini.union("-", " ")).ques + self.DELETE_SPACE + insert(" ")
        self.tagger = self.add_tokens(country.ques + self.tag_field("number_part", self.number_part))
        country = self.verbalize_field("country_code", self.country_code) + delete(" ") + insert(" ")
        self.verbalizer = self.delete_tokens(country.ques + self.verbalize_field("number_part", self.number_part))
