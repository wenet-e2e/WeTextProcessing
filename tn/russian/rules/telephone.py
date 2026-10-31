# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
from tn.russian.rules.utils import DIGIT, number_names


TOKEN_ORDER = {}


class Telephone(Processor):
    """Phone numbers, e.g. "8-913-983-56-01" -> telephone { country_code: "8" number_part: "913-983-56-01" }."""

    def __init__(self):
        super().__init__("telephone", token_orders=TOKEN_ORDER)
        number = number_names().nominative
        separator = pynini.cross("-", " ")
        country_code = pynini.closure(pynutil.add_weight(pynutil.delete("+"), 0.1), 0, 1) + number
        last = pynini.closure(pynini.cross("0", "ноль ")) + number
        number_part = (
            DIGIT ** 3 @ number + separator + DIGIT ** 3 @ number + separator
            + DIGIT ** 2 @ number + separator + DIGIT ** 2 @ last
        ).optimize()

        country_tagger = self.tag_field("country_code", country_code) + pynutil.delete("-") + pynutil.insert(" ")
        self.tagger = self.add_tokens(country_tagger.ques + self.tag_field("number_part", number_part))
        country_verbalizer = self.verbalize_field("country_code", country_code) + pynini.accep(" ")
        self.verbalizer = self.delete_tokens(
            country_verbalizer.ques + self.verbalize_field("number_part", number_part))
