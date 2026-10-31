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

from itn.russian.rules.utils import DIGIT, number_names
from tn.processor import Processor


class Telephone(Processor):
    """Spoken phone numbers grouped 3-3-2-2, e.g. "восемь девятьсот тринадцать ..." -> "8-913-983-56-01"."""

    def __init__(self):
        super().__init__(name="telephone", ordertype="itn")
        number = number_names().nominative
        separator = pynini.cross("-", " ")
        # Written -> spoken readings as in TN, inverted below.
        country_code = pynini.closure(pynutil.add_weight(pynutil.delete("+"), 0.1), 0, 1) + number
        last = pynini.closure(pynini.cross("0", "ноль ")) + number
        number_part = (
            DIGIT ** 3 @ number + separator + DIGIT ** 3 @ number + separator
            + DIGIT ** 2 @ number + separator + DIGIT ** 2 @ last
        )
        self.country_code = country_code.invert().optimize()
        self.number_part = number_part.invert().optimize()

        country_tagger = self.tag_field("country_code", self.country_code) + pynutil.delete(" ") + pynutil.insert(" ")
        self.tagger = self.add_tokens(country_tagger.ques + self.tag_field("number_part", self.number_part))
        country_verbalizer = self.verbalize_field("country_code", self.country_code)
        country_verbalizer += pynutil.delete(" ") + pynutil.insert("-")
        self.verbalizer = self.delete_tokens(
            country_verbalizer.ques + self.verbalize_field("number_part", self.number_part))
