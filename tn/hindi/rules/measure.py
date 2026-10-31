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

from tn.hindi.rules.utils import HI_DEDH, HI_DHAI, HI_PAUNE, HI_SADHE, HI_SAVVA, MINUS, hindi_path
from tn.processor import Processor

TOKEN_ORDER = {}


class Measure(Processor):
    """``१२.५ kg`` -> ``measure { integer_part: "१२" fractional_part: "५" units: "kg" }``.

    Numbers and units stay raw; the verbalizer chooses idioms such as ``1.5 yr`` -> डेढ़ साल,
    and the ``yr`` reading (वर्ष/साल) from the written number.
    """

    def __init__(self, cardinal, decimal):
        super().__init__("measure", token_orders=TOKEN_ORDER)
        number = pynini.union(
            cardinal.zero, cardinal.digit, cardinal.teens_and_ties, cardinal.graph_hundreds,
            cardinal.graph_thousands, cardinal.graph_ten_thousands, cardinal.graph_lakhs,
            cardinal.graph_ten_lakhs).optimize()
        large = cardinal.graph_without_leading_zeros
        small = (cardinal.zero | cardinal.digit | cardinal.teens_and_ties | cardinal.graph_hundreds).optimize()
        fractional = decimal.fractional_graph
        units = pynini.string_file(hindi_path("measure/unit.tsv"))
        units = (pynini.difference(units.copy().project("input"), "yr") @ units).optimize()
        quarterly = pynini.union(
            pynini.string_file(hindi_path("measure/quarterly_units_map.tsv")),
            pynini.string_file(hindi_path("measure/quarterly_units_list.tsv"))).optimize()
        year_formal = pynini.string_file(hindi_path("measure/unit_year_formal.tsv"))
        year_informal = pynini.cross("yr", "साल")
        paune = pynini.string_file(hindi_path("whitelist/paune_mappings.tsv"))

        sign_tag = self.tag_field("negative", pynini.accep("-")) + insert(" ")
        sign_verb = self.verbalize_field("negative", pynini.cross("-", MINUS)) + delete(" ")
        # "-" before a cardinal reads "minus" in NeMo; before a decimal it reads ऋणात्मक.
        cardinal_sign_verb = self.verbalize_field("negative", pynini.cross("-", "minus ")) + delete(" ")

        def unit_pair(graph):
            tagger = self.DELETE_SPACE + insert(" ") + self.tag_field("units", graph)
            verbalizer = insert(" ") + self.verbalize_field("units", graph, leading_space=True)
            return tagger, verbalizer

        def integer(graph):
            return self.tag_field("integer", graph), self.verbalize_field("integer", graph)

        def decimal_number(integer_graph, fraction_graph):
            tagger = (self.tag_field("integer_part", self.input_projection(integer_graph)) + delete(".")
                      + insert(" ") + self.tag_field("fractional_part", self.input_projection(fraction_graph)))
            verbalizer = (self.verbalize_field("integer_part", integer_graph)
                          + self.verbalize_field("fractional_part", fraction_graph, leading_space=True))
            return tagger, verbalizer

        unit_t, unit_v = unit_pair(units)
        quarter_t, quarter_v = unit_pair(quarterly)
        formal_t, formal_v = unit_pair(year_formal)
        informal_t, informal_v = unit_pair(year_informal)
        int_t, int_v = integer(number)
        large_t, large_v = integer(large)
        small_t, small_v = integer(small)
        dec_t, dec_v = decimal_number(number + insert(" दशमलव "), fractional)

        # Quarter idioms read the written "integer.fraction" pair as one spoken number.
        five = delete(pynini.union("५", "5"))
        quarter_numbers = [
            (decimal_number(pynini.cross(pynini.union("1", "१"), HI_DEDH), five), -0.2),
            (decimal_number(pynini.cross(pynini.union("2", "२"), HI_DHAI), five), -0.2),
            (decimal_number(insert(HI_SAVVA + " ") + number, delete(pynini.union("२५", "25"))), -0.1),
            (decimal_number(insert(HI_SADHE + " ") + number, five), -0.1),
            (decimal_number(insert(HI_PAUNE + " ") + paune, delete(pynini.union("७५", "75"))), -0.5),
        ]

        patterns = [
            (sign_tag.ques + dec_t + unit_t, sign_verb.ques + dec_v + unit_v, 0.1),
            (sign_tag.ques + dec_t + formal_t, sign_verb.ques + dec_v + formal_v, 0.1),
            (sign_tag.ques + int_t + unit_t, cardinal_sign_verb.ques + int_v + unit_v, 0.1),
            (sign_tag.ques + large_t + formal_t, cardinal_sign_verb.ques + large_v + formal_v, 0.1),
            (sign_tag.ques + small_t + informal_t, cardinal_sign_verb.ques + small_v + informal_v, -0.1),
        ]
        for (number_t, number_v), weight in quarter_numbers:
            patterns.append((sign_tag.ques + number_t + quarter_t, cardinal_sign_verb.ques + number_v + quarter_v,
                             weight))
        # "200x10" -> दो सौ बाई दस.
        by = pynini.union("x", "X", "*")
        dimension_t = (sign_tag.ques + int_t + insert(" ") + self.tag_field("units", by) + insert(" ")
                       + (sign_tag.ques + self.tag_field("multiplier", number)))
        dimension_v = (cardinal_sign_verb.ques + int_v + insert(" ")
                       + self.verbalize_field("units", pynini.cross(by, "बाई"), leading_space=True) + insert(" ")
                       + cardinal_sign_verb.ques + self.verbalize_field("multiplier", number, leading_space=True))
        # NeMo emits a measure plus a separate cardinal token; keep that token's 1.1 classifier cost.
        patterns.append((dimension_t, dimension_v, 0.1 + 1.1))

        self.tagger = self.add_tokens(pynini.union(*[add_weight(t, w) for t, _, w in patterns]))
        self.verbalizer = self.delete_tokens(pynini.union(*[add_weight(v, w) for _, v, w in patterns]))
