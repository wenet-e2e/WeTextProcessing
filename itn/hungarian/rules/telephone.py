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
from pynini.lib.pynutil import delete

from itn.hungarian.rules.utils import DIGIT, NONZERO, data_keys
from tn.processor import Processor

TOKEN_ORDER = {}


class Telephone(Processor):
    """Phone numbers and IPv4 addresses, e.g. ``plusz harminchat egy ...`` -> ``+36 1 4888 700``.

    A number starts with a country code (``plusz harminchat``) or the
    domestic trunk prefix (``nulla hat``); raw spoken ``country_code`` and
    ``number_part`` fields are written as space-separated digit groups.
    """

    def __init__(self, cardinal):
        super().__init__("telephone", ordertype="itn", token_orders=TOKEN_ORDER)
        digits = pynini.invert(cardinal.spoken)
        zero = pynini.cross("nulla", "0") + delete(" ")
        digit = digits @ NONZERO
        two = (digits @ (NONZERO + DIGIT) | zero + digit).optimize()
        tens = digits @ (NONZERO + DIGIT)
        three = (digits @ (NONZERO + DIGIT ** 2) | zero + tens | zero + zero + digit).optimize()
        four = (digits @ (NONZERO + DIGIT ** 3) | zero + three).optimize()
        up_to_three = (digits @ pynini.closure(DIGIT, 1, 3)).optimize()

        space = pynini.accep(" ")
        area = digits @ data_keys("telephone/area_codes.tsv")
        base = pynini.union(
            three + space + three,
            three + space + two + space + two,
            three + space + four,
            two + space + four,
            four + space + three,
            two + space + two + space + three,
        )
        trunk = pynini.cross("nulla hat", "06")
        number = area + space + base
        ip = up_to_three + (pynini.cross(" pont ", ".") + up_to_three) ** 3
        self.number = (number | ip).optimize()
        self.trunk_number = (trunk + space + number).optimize()
        country = digits @ data_keys("telephone/country_codes.tsv")
        self.country = (pynini.cross("plusz ", "+") + country).optimize()

        country_t = self.tag_field("country_code", self.country) + space
        number_t = self.tag_field("number_part", self.number)
        trunk_t = self.tag_field("number_part", self.trunk_number)
        self.tagger = self.add_tokens(country_t + number_t | trunk_t | self.tag_field("number_part", ip))

        country_v = self.verbalize_field("country_code", self.country) + space
        number_v = self.verbalize_field("number_part", self.number | self.trunk_number)
        self.verbalizer = self.delete_tokens(country_v.ques + number_v)
