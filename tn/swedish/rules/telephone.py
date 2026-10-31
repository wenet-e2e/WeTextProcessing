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
from tn.swedish.rules.utils import tsv

TOKEN_ORDER = {}


class Telephone(Processor):
    """Raw prompts, Swedish phone numbers, IP addresses and SSNs."""

    def __init__(self, cardinal, deterministic=True):
        super().__init__("telephone", token_orders=TOKEN_ORDER)
        self.deterministic = deterministic
        ins = pynutil.insert(" ")
        separator = pynutil.insert(", ")
        zero_space = cardinal.zero_space
        digit = cardinal.digit
        two = cardinal.two_digits_read
        three = cardinal.three_digits_read
        one_to_three = (digit | two | three).optimize()

        abbr = tsv("telephone/telephone_abbr") + pynutil.delete(":").ques
        prompt = tsv("telephone/telephone_prompt")
        inner = prompt | abbr | prompt + " " + abbr
        country = pynini.cross("+", "plus ").ques + one_to_three
        country_code = (inner | country | inner + " " + country).optimize()

        space = pynini.accep(" ")
        area = (pynini.cross("(0)", "noll ") | zero_space) + one_to_three + pynutil.delete("-").ques + separator
        area |= pynutil.delete("(") + one_to_three + pynutil.delete(")") + separator
        base = pynini.union(
            three + space + three + space + two,
            three + space + two + space + two,
            three + space + two + ins + two,
            two + space + two + space + two,
            two + space + two + ins + two,
            three + space + two,
        )
        number = (area + pynutil.delete(pynini.accep(" ").star) + base).optimize()
        special = tsv("telephone/special_numbers")
        extension = (
            pynini.cross(pynini.union("ankn", "ankn.", "anknytning"), "anknytning")
            + (space | ins) + one_to_three
        ).optimize()

        ip = (one_to_three + (pynini.cross(".", " punkt ") + one_to_three)**3).optimize()
        four = (digit | cardinal.zero) + (ins + (digit | cardinal.zero))**3
        ssn = (three + pynini.cross("-", ", ") + two + pynini.cross("-", ", ") + four).optimize()

        country_t = self.tag_field("country_code", country_code)
        number_t = self.tag_field("number_part", number)
        extension_t = space + self.tag_field("extension", extension)
        optional_space = space | ins
        tagger = (
            country_t + optional_space + number_t + extension_t.ques
            | number_t + extension_t
            | country_t + optional_space + self.tag_field("number_part", special)
            | (self.tag_field("country_code", tsv("telephone/ip_prompt")) + space).ques
            + self.tag_field("number_part", ip)
            | (self.tag_field("country_code", tsv("telephone/ssn_prompt")) + space).ques
            + self.tag_field("number_part", ssn)
        )
        self.tagger = self.add_tokens(tagger)

        prompts = country_code | tsv("telephone/ip_prompt") | tsv("telephone/ssn_prompt")
        numbers = number | special | ip | ssn
        verbalizer = (
            (self.verbalize_field("country_code", prompts) + pynutil.delete(" ") + ins).ques
            + self.verbalize_field("number_part", numbers.optimize())
            + (ins + self.verbalize_field("extension", extension, leading_space=True)).ques
        )
        self.verbalizer = self.delete_tokens(verbalizer)
