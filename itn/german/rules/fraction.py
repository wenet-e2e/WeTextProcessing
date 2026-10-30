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


class Fraction(Processor):
    """Capture each spoken operand independently, including mixed fractions."""

    def __init__(self, cardinal, tn_fraction):
        super().__init__("fraction", ordertype="itn", token_orders=TOKEN_ORDER)
        omit_spaces = pynini.closure(self.VSIGMA | pynutil.delete(" "))
        # The source converts every operand through graph_no_exception,
        # including its 0.001 cost. Retain that cost so an ambiguous mixed
        # parse does not outrank the same words read as a simple fraction.
        integer = pynutil.add_weight((tn_fraction.integer_graph @ omit_spaces).invert().optimize(), 0.001)
        numerator = pynutil.add_weight((tn_fraction.numerator_graph @ omit_spaces).invert().optimize(), 0.001)
        denominator = pynutil.add_weight((tn_fraction.denominator_graph @ omit_spaces).invert().optimize(), 0.001)
        not_two = denominator @ (self.DIGIT.plus - "2")
        not_one = numerator @ (self.DIGIT.plus - "1")
        half = pynutil.add_weight(pynini.cross("halb", "2"), 0.001)
        halves = denominator @ pynini.accep("2")
        one = numerator @ pynini.accep("1")

        separator = pynutil.delete(" ") + pynutil.insert(" ")
        fraction_tagger = (
            self.tag_field("numerator", numerator) + separator + self.tag_field("denominator", not_two)
            | self.tag_field("numerator", not_one) + separator + self.tag_field("denominator", halves)
            | self.tag_field("numerator", one) + separator + self.tag_field("denominator", half)
        )
        fraction_verbalizer = (
            self.verbalize_field("numerator", numerator) + pynutil.insert("/")
            + self.verbalize_field("denominator", not_two, leading_space=True)
            | self.verbalize_field("numerator", not_one) + pynutil.insert("/")
            + self.verbalize_field("denominator", halves, leading_space=True)
            | self.verbalize_field("numerator", one) + pynutil.insert("/")
            + self.verbalize_field("denominator", half, leading_space=True)
        )
        conjunction = pynutil.add_weight(pynutil.delete("und"), 0.001)
        integer_tagger = pynini.closure(
            self.tag_field("integer_part", integer) + separator
            + pynini.closure(self.tag_field("conjunction", conjunction) + separator, 0, 1),
            0, 1,
        )
        integer_verbalizer = pynini.closure(
            self.verbalize_field("integer_part", integer) + pynutil.delete(" ") + pynutil.insert(" ")
            + pynini.closure(
                self.verbalize_field("conjunction", conjunction) + pynutil.delete(" "), 0, 1,
            ),
            0, 1,
        )
        sign_tagger = pynini.closure(
            self.tag_field("negative", cardinal.sign) + separator, 0, 1,
        )
        sign_verbalizer = pynini.closure(
            self.verbalize_field("negative", cardinal.sign) + pynutil.delete(" "), 0, 1,
        )
        self.fields_tagger = (sign_tagger + integer_tagger + fraction_tagger).optimize()
        self.fields_verbalizer = (sign_verbalizer + integer_verbalizer + fraction_verbalizer).optimize()
        self.tagger = self.add_tokens(self.fields_tagger)
        self.verbalizer = self.delete_tokens(self.fields_verbalizer)
