# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
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
from pynini.lib.pynutil import add_weight, insert

from itn.vietnamese.rules.cardinal import DELETE_SPACE, numbers_file
from tn.processor import Processor


class Telephone(Processor):
    """Digit-by-digit numbers keep their words raw: phone numbers, IPv4 addresses and card numbers."""

    def __init__(self, cardinal):
        super().__init__(name="telephone", ordertype="itn", token_orders={})
        ds = DELETE_SPACE
        digit = (cardinal.digit | cardinal.zero).optimize()
        last_digit = digit | numbers_file("digit_special")
        two_digits = cardinal.graph_no_exception @ self.DIGIT**2

        mobile = pynini.cross("không", "0") + ds + pynini.closure(digit + ds, 8) + last_digit
        mobile @= "0" + self.DIGIT**2 + insert("-") + self.DIGIT**3 + insert("-") + self.DIGIT**4
        phone = add_weight(mobile, -0.01) | pynini.closure(digit + ds, 2) + last_digit
        octet = pynini.closure(digit + ds, 0, 2) + digit | two_digits
        ip = octet + (ds + pynini.cross("chấm", ".") + ds + octet) ** 3
        card16 = pynini.closure(digit + ds, 15) + digit
        card16 @= self.DIGIT**4 + (insert(" ") + self.DIGIT**4) ** 3
        card15 = pynini.closure(digit + ds, 14) + digit
        card15 @= self.DIGIT**4 + insert(" ") + self.DIGIT**6 + insert(" ") + self.DIGIT**5
        self.number_graph = (
            add_weight(ip, -0.01) | add_weight(card16 | card15, -0.005) | phone
        ).optimize()
        self.phone_graph = phone.optimize()
        country_digits = add_weight(two_digits, -0.001) | digit + ds + digit | digit
        self.country_graph = (pynini.cross("cộng ", "+") + country_digits).optimize()

        country = self.tag_field("country_code", self.country_graph) + ds + insert(" ")
        country += self.tag_field("number_part", self.phone_graph)
        self.tagger = self.add_tokens(add_weight(country, -0.1) | self.tag_field("number_part", self.number_graph))

        country_v = self.verbalize_field("country_code", self.country_graph) + pynini.accep(" ")
        country_v += self.verbalize_field("number_part", self.phone_graph)
        number_v = self.verbalize_field("number_part", self.number_graph)
        self.verbalizer = self.delete_tokens(country_v | number_v)
