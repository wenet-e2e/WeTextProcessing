# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pynini.lib import pynutil

from tn.processor import Processor
from tn.spanish.rules.utils import ALPHA, ONES, data_file, rewrite

TOKEN_ORDER = {"telephone": ["country_code", "number_part", "extension"]}


class Telephone(Processor):
    """Raw country code, number and extension; V reads the digits one by one."""

    def __init__(self, deterministic=True):
        super().__init__("telephone", token_orders=TOKEN_ORDER)
        self.deterministic = deterministic

        def table(name):
            return pynini.invert(data_file("numbers/" + name + ".tsv"))

        single = rewrite(table("digit"), pynini.cross(ONES, "uno")) | table("zero")
        double = table("teen") | table("ties") + (pynutil.delete("0") | pynutil.insert(" y ") + single)
        double = (double | table("twenties")).optimize()
        space = pynutil.insert(" ")
        separators = pynini.union("-", " ", ".")
        delete_separator = pynini.closure(pynutil.delete(separators), 0, 1)

        def digits(count):
            return pynini.closure(single + space, count - 1, count - 1) + single

        triple = digits(3)
        country_code = pynini.closure(pynini.cross("+", "más "), 0, 1) + (single | double | triple)
        prompts = data_file("telephone/telephone_prompt.tsv")
        country_code_graph = (
            country_code | data_file("telephone/ip_prompt.tsv")
            | prompts + pynini.closure(pynini.accep(" ") + country_code, 0, 1)
        ).optimize()

        group = single + pynini.closure(space + single, 0, 2)
        ip_graph = group + (pynini.cross(".", " punto ") + group) ** 3
        area_code = (
            pynini.closure(pynutil.delete("("), 0, 1) + digits(3) + pynini.closure(pynutil.delete(")"), 0, 1)
        )
        ten = area_code + delete_separator + space + digits(3) + delete_separator + space + digits(4)
        nine = area_code + delete_separator + space + digits(3) + delete_separator + space + digits(3)
        eight = pynini.closure(area_code + space, 0, 1) + delete_separator + digits(4) + delete_separator + space + digits(4)
        if not deterministic:
            pair = single + space + double + space + delete_separator
            ten |= pair + pair + double + space + double
            nine |= pair + pair + single + space + double
            eight |= double + space + double + space + delete_separator + double + space + double
        # Numbers spelled with letters: "1-800-go-u-haul".
        letters_area = area_code @ pynini.cross("ocho cero cero", "ochocientos")
        word = pynini.closure(single | ALPHA, 1, 8)
        words = pynini.closure(word + pynini.cross(separators, " "), 0, 2) + word
        letters = pynutil.add_weight(letters_area + delete_separator + space + words, 0.01)
        number_graph = pynini.union(ten, nine, eight, letters, ip_graph).optimize()
        extension_graph = (
            pynini.closure(pynutil.delete("ext."), 0, 1) + delete_separator + pynutil.insert("extensión ")
            + pynini.closure(single + space, 0, 3) + single
        ).optimize()

        # Tagger: separators between the parts are structure only.
        country_t = self.tag_field("country_code", country_code_graph) + delete_separator + space
        number_t = self.tag_field("number_part", number_graph)
        extension_t = pynutil.add_weight(
            pynutil.delete(separators) + space + self.tag_field("extension", extension_graph), 0.01,
        )
        self.tagger = self.add_tokens(
            pynini.closure(country_t, 0, 1) + number_t + pynini.closure(extension_t, 0, 1)
        )

        number_v = self.verbalize_field("number_part", number_graph, leading_space=True)
        graph = (
            self.verbalize_field("country_code", country_code_graph) + space + number_v
            | self.verbalize_field("number_part", number_graph)
        )
        graph += pynini.closure(space + self.verbalize_field("extension", extension_graph, leading_space=True), 0, 1)
        self.verbalizer = self.delete_tokens(graph)
