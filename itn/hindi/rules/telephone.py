# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from itn.hindi.rules.utils import CHAR, DELETE_SPACE, WHITE_SPACE, spoken_to_written, string_file
from tn.processor import Processor

TOKEN_ORDER = {}


class Telephone(Processor):
    """``प्लस नौ एक नौ आठ सात छह पाँच चार तीन दो एक शून्य`` -> ``telephone { value: "..." }`` -> ``+९१ ९८७६५४३२१०``.

    Context words (``मोबाइल``, ``पिन`` ...) up to five words away are kept verbatim.
    """

    def __init__(self):
        super().__init__("telephone", ordertype="itn", token_orders=TOKEN_ORDER)
        zero = spoken_to_written("numbers/zero.tsv") | spoken_to_written("telephone/eng_zero.tsv")
        non_zero = spoken_to_written("numbers/digit.tsv") | spoken_to_written("telephone/eng_digit.tsv")
        digit = (zero | non_zero).optimize()
        cues = string_file("telephone/context_cues.tsv")

        def context(kind):
            keywords = (delete(kind) @ cues).project("output").optimize()
            # The upstream digit-word exclusion only removes single-character words, i.e. nothing.
            word = (CHAR - WHITE_SPACE).plus + WHITE_SPACE
            window = pynini.closure(word, 0, 5)
            return (keywords + window).optimize(), (window + keywords).optimize()

        def digits(first, length):
            return first + pynini.closure(DELETE_SPACE + digit, length - 1, length - 1)

        def with_context(kind, number):
            before, after = context(kind)
            return before + WHITE_SPACE + number | number + WHITE_SPACE + after | number

        before, after = context("mobile")
        country_code = (pynini.cross("प्लस", "+") + pynini.closure(DELETE_SPACE + digit, 2, 2)
                        + delete(WHITE_SPACE) + insert(" "))
        country_code = (before + WHITE_SPACE).ques + country_code
        mobile = digits(non_zero, 10) + (WHITE_SPACE + after).ques
        mobile = country_code.ques + mobile
        landline = with_context("landline", digits(zero, 11))
        credit = with_context("credit", digits(digit, 4))
        pincode = with_context("pincode", digits(digit, 6))
        graph = (add_weight(mobile, 0.7) | add_weight(landline, 0.8) | add_weight(credit, 0.9)
                 | add_weight(pincode, 1))
        self.graph = graph.optimize()
        self.tagger = self.add_tokens(self.tag_field("value", self.graph))
        self.verbalizer = self.delete_tokens(self.verbalize_field("value", self.graph))
