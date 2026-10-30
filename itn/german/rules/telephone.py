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


TOKEN_ORDER = {}


class Telephone(Processor):
    def __init__(self, tn_cardinal):
        super().__init__("telephone", ordertype="itn", token_orders=TOKEN_ORDER)
        digit = ((self.DIGIT - "0") @ tn_cardinal.two_digits).invert().optimize()
        zero = pynini.cross("null", "0")
        self.digits = (digit | zero).optimize()
        group = (digit + pynutil.delete(" ")) ** 3 + digit
        self.country_code_graph = (
            pynutil.insert("(") + zero + pynutil.delete(" ")
            + (digit + pynutil.delete(" ")) ** 2 + digit + pynutil.insert(")")
        ).optimize()
        self.number_part_graph = (
            group + pynutil.delete(" ") + pynutil.insert("-") + group
        ).optimize()
        self.graph = (self.country_code_graph + pynini.accep(" ") + self.number_part_graph).optimize()
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
