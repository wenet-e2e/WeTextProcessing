# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from tn.hungarian.rules.utils import data_keys, data_map
from tn.processor import Processor

TOKEN_ORDER = {"telephone": ["prompt", "country_code", "number_part"]}


class Telephone(Processor):
    """Phone numbers and IPv4 addresses, e.g. ``+36-80/555-111``, ``06 40 49 4949``.

    Raw fields: an optional ``prompt`` (``tel.``), ``country_code`` and
    ``number_part``; digit groups are read as numbers.
    """

    def __init__(self, cardinal):
        super().__init__("telephone", token_orders=TOKEN_ORDER)
        digit = cardinal.digit
        two = cardinal.two_digits_read
        three = cardinal.three_digits_read
        four = cardinal.four_digits_read
        up_to_three = (digit | two | three).optimize()

        separator = pynini.union(" ", pynini.cross("-", " "))
        area_separator = separator | pynini.cross("/", " ")
        area = data_keys("telephone/area_codes.tsv") @ cardinal.plain
        base = pynini.union(
            three + separator + three,
            three + separator + two + separator + two,
            three + separator + four,
            two + separator + four,
            four + separator + three,
            two + separator + two + separator + three,
        )
        trunk = pynini.cross("06", "nulla hat")
        trunk |= delete("(") + trunk + delete(")")
        number = area + area_separator + base
        ip = up_to_three + (pynini.cross(".", " pont ") + up_to_three) ** 3
        self.number = ((trunk + separator).ques + number | ip).optimize()

        plus = pynini.cross("+", "plusz ") | pynini.cross("00", "nulla nulla ")
        country = data_keys("telephone/country_codes.tsv") @ cardinal.plain
        country = delete("(").ques + country + delete(")").ques
        self.country = (plus + delete(" ").ques + country).optimize()
        self.prompt = data_map("telephone/telephone_abbr.tsv")

        number_t = self.tag_field("number_part", self.number)
        country_t = self.tag_field("country_code", self.country) + delete(pynini.union(" ", "-")) + insert(" ")
        prompt_t = self.tag_field("prompt", self.prompt) + delete(":").ques + pynini.accep(" ")
        self.tagger = self.add_tokens(prompt_t.ques + country_t.ques + number_t)

        prompt_v = self.verbalize_field("prompt", self.prompt) + pynini.accep(" ")
        country_v = self.verbalize_field("country_code", self.country) + pynini.accep(" ")
        number_v = self.verbalize_field("number_part", self.number)
        self.verbalizer = self.delete_tokens(prompt_v.ques + country_v.ques + number_v)
