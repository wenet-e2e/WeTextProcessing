# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from tn.armenian.rules.utils import preferred_map
from tn.processor import Processor


class Cardinal(Processor):
    """Cardinals, e.g. ``55`` -> ``հիսունհինգ``.

    The tagger keeps the written digits (including optional group spaces such
    as ``8 555``); the verbalizer reads them with Armenian numerals.
    """

    def __init__(self):
        super().__init__("cardinal", token_orders={})
        self.build_graphs()
        self.tagger = self.add_tokens(self.tag_field("integer", self.graph))
        self.verbalizer = self.delete_tokens(self.verbalize_field("integer", self.graph))

    def build_graphs(self):
        digit = self.DIGIT
        zero = pynini.cross("0", "զրո")
        digits = preferred_map("armenian/data/numbers/digit.tsv", invert=True)
        digits_no_one = (digit - "1") @ digits
        ties = preferred_map("armenian/data/numbers/ties.tsv", invert=True)
        double_digits = (pynini.cross("1", "տասը") | ties) + delete("0")
        double_digits |= (pynini.cross("1", "տասն") | ties) + digits
        self.double_digits = double_digits.optimize()
        self.one_to_all_tens = (digits | double_digits).optimize()

        hundreds_parts = delete("0") + self.INSERT_SPACE + digits | self.INSERT_SPACE + double_digits
        one_hundreds = pynini.cross("1", "հարյուր") + (delete("00") | hundreds_parts)
        multiple_hundreds = digits_no_one + insert(" հարյուր") + (delete("00") | hundreds_parts)
        all_hundreds = (one_hundreds | multiple_hundreds).optimize()

        # Group separators are optional spaces after the thousands digits.
        separator = delete(self.SPACE).star
        one_thousand = pynini.cross("1", "հազար") + separator
        other_thousands = (digits_no_one | double_digits | all_hundreds) + insert(" հազար") + separator
        thousands = one_thousand | other_thousands
        all_thousands = (
            thousands + delete("000")
            | thousands + delete("00") + self.INSERT_SPACE + digits
            | thousands + delete("0") + self.INSERT_SPACE + double_digits
            | thousands + self.INSERT_SPACE + all_hundreds
        ).optimize()

        to_hundreds = digits | double_digits | all_hundreds
        to_thousands = to_hundreds | all_thousands
        components = (separator + pynini.closure(digit, 3)).closure(2)
        rest = delete("0").closure(0, 6) + (self.INSERT_SPACE + to_thousands).ques
        all_millions = (to_hundreds + insert(" միլիոն") + (components @ rest)).optimize()

        to_millions = to_thousands | all_millions
        components = (separator + pynini.closure(digit, 3)).closure(3)
        rest = delete("0").closure(0, 9) + (self.INSERT_SPACE + to_millions).ques
        all_billions = (to_hundreds + insert(" միլիարդ") + (components @ rest)).optimize()

        self.graph = (zero | to_millions | all_billions).optimize()
