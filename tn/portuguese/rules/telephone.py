# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pynini.lib.pynutil import add_weight, delete, insert

from tn.processor import Processor
from tn.utils import get_abs_path


TOKEN_ORDER = {}


class Telephone(Processor):
    """Raw Brazilian phone numbers, IP addresses and extensions, read digit by digit."""

    def __init__(self, cardinal):
        super().__init__("telephone", token_orders=TOKEN_ORDER)
        digit = pynini.string_map([(str(n), cardinal.digits[str(n)]) for n in range(1, 10)] + [("0", cardinal.zero)])
        space = insert(" ")
        separator = delete(pynini.union("-", "."))
        spaces = delete(self.SPACE).star

        def digits(count):
            return digit + (space + digit) ** (count - 1)

        country = pynini.cross("+", "mais ") + (digits(1) | digits(2) | digits(3))
        ip_prompt = pynini.string_file(get_abs_path("portuguese/data/telephone/ip_prompt.tsv"))
        phone_prompt = pynini.string_file(get_abs_path("portuguese/data/telephone/telephone_prompt.tsv"))
        prompt = country | ip_prompt | phone_prompt + (pynini.accep(" ") + country).ques
        self.country_graph = (prompt + separator.ques).optimize()

        area = delete("(") + digits(2) + delete(")") | digits(2)
        local = (
            area + spaces + space + digits(5) + separator + space + digits(4)
            | area + spaces + space + digits(4) + separator + space + digits(4)
            | digits(5) + separator + space + digits(4)
            | digits(4) + separator + space + digits(4)
            | digits(3) + separator + space + digits(4)
        )
        block = digit + (space + digit).ques + (space + digit).ques
        ip = block + (pynini.cross(".", " ponto ") + block) ** 3
        self.number_graph = (local | add_weight(ip, 0.01)).optimize()
        extension = (
            delete("ramal") + spaces + insert("ramal ")
            | (delete("extensão") | delete("ext.")) + spaces + insert("extensão ")
        )
        self.extension_graph = (extension + digit + (space + digit) ** (0, 3)).optimize()

        self.tagger = self.add_tokens(
            (self.tag_field("country_code", self.country_graph) + spaces + insert(" ")).ques
            + self.tag_field("number_part", self.number_graph)
            + (spaces + insert(" ") + self.tag_field("extension", self.extension_graph)).ques
        )
        self.verbalizer = self.delete_tokens(
            (self.verbalize_field("country_code", self.country_graph) + delete(" ") + insert(" ")).ques
            + self.verbalize_field("number_part", self.number_graph)
            + (insert(" ") + self.verbalize_field("extension", self.extension_graph, leading_space=True)).ques
        )
