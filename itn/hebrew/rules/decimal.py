# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

from itn.hebrew.rules.cardinal import GROUP_BY_THREES
from itn.hebrew.rules.utils import DELETE_SPACE, PREFIX, SPACE, accep_file, first_field, next_field, string_file, trim
from tn.processor import Processor

TOKEN_ORDER = {"decimal": ["prefix", "negative", "integer_part", "fractional_part", "fraction", "quantity"]}


class Decimal(Processor):
    """Spoken decimals, e.g. ``ארבע נקודה שמונים וחמש מיליארד`` -> ``4.85 מיליארד``.

    Digits after ``נקודה`` stay in ``fractional_part``; a spoken fraction
    joined with ``ו`` (``וחצי``, ``ושלושת רבעי``) stays in ``fraction``.
    """

    def __init__(self, cardinal):
        super().__init__(name="decimal", ordertype="itn", token_orders=TOKEN_ORDER)
        sep = delete(SPACE.plus) + insert(" ")
        fractions = string_file("numbers/decimal_fractions.tsv")
        minutes = accep_file("decimals/minutes_exception.tsv")
        fractions_wo_minutes = ((pynini.project(fractions, "input") - minutes) @ fractions).optimize()
        self.fraction = (delete("ו") + fractions).optimize()
        fraction_wo_minutes = (delete("ו") + fractions_wo_minutes).optimize()

        digits = string_file("numbers/zero.tsv") | cardinal.graph_two_digit
        # Fewer spoken groups win: ``שתיים עשרה`` is 12, not 2 and 10.
        self.fractional = trim((digits + DELETE_SPACE + add_weight(insert(""), 0.01)).star + digits)

        negative = self.tag_field("negative", pynini.accep("מינוס")) + sep
        integer = self.tag_field("integer_part", cardinal.graph) + sep
        point = delete("נקודה") + delete(SPACE.plus)
        with_point = integer.ques + point + self.tag_field("fractional_part", self.fractional)
        # Without ``נקודה`` an hour-like integer with a minute-like fraction
        # (``שתיים וחצי``) is left to the time grammar unless negated.
        wo_point = (integer + self.tag_field("fraction", fraction_wo_minutes)
                    | self.tag_field("integer_part", cardinal.graph_wo_viable_hours) + sep
                    + self.tag_field("fraction", self.fraction))
        any_fraction = integer + self.tag_field("fraction", self.fraction)
        self.number = (with_point | any_fraction).optimize()
        quantity = pynini.union("מיליון", "מיליארד")
        quantified = (self.tag_field("integer_part", cardinal.graph_hundred) + sep
                      + self.tag_field("quantity", quantity)
                      | self.number + sep + self.tag_field("quantity", quantity | "אלף"))
        tagger = negative.ques + (with_point | wo_point | quantified) | negative + any_fraction
        self.tagger = self.add_tokens((self.tag_field("prefix", PREFIX) + insert(" ")).ques + tagger)

        integer = cardinal.graph @ GROUP_BY_THREES
        fraction = insert(".") + self.fraction
        fractional = insert(".") + self.fractional
        number = (first_field(self, "integer_part", integer).ques + self.verbalize_field("fractional_part", fractional)
                  | first_field(self, "integer_part", integer) + self.verbalize_field("fraction", fraction))
        self.verbalize_number = number.optimize()
        number = (number | self.verbalize_field("integer_part", integer)) + next_field(
            self, "quantity", insert(" ") + (quantity | "אלף")).ques
        self.verbalizer = self.delete_tokens(
            first_field(self, "prefix", PREFIX + insert("-")).ques
            + first_field(self, "negative", pynini.cross("מינוס", "-")).ques + number)
