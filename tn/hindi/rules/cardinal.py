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

from tn.hindi.rules.utils import ALL_DIGIT, ALL_ZERO, hindi_path
from tn.processor import Processor

TOKEN_ORDER = {}

# Indian place-value names from hundreds upward; each scale spans two digits.
SCALES = ["हज़ार", "लाख", "करोड़", "अरब", "खरब", "नील", "पद्म", "शंख"]


class Cardinal(Processor):
    """Written digits stay raw in the tagger; the verbalizer reads them in Hindi.

    ``२३`` is tagged as ``cardinal { integer: "२३" }`` and verbalized as ``तेईस``.
    """

    def __init__(self):
        super().__init__("cardinal", token_orders=TOKEN_ORDER)
        self.build_graphs()
        negative = pynini.accep("-")
        self.tagger = self.add_tokens(
            (self.tag_field("negative", negative) + insert(" ")).ques + self.tag_field("integer", self.final_graph))
        # NeMo's Hindi cardinal verbalizer keeps the English "minus" prefix.
        sign = self.verbalize_field("negative", pynini.cross("-", "minus ")) + delete(" ")
        self.verbalizer = self.delete_tokens(sign.ques + self.verbalize_field("integer", self.final_graph))

    def build_graphs(self):
        digit = pynini.string_file(hindi_path("numbers/digit.tsv"))
        zero = pynini.string_file(hindi_path("numbers/zero.tsv"))
        teens_ties = pynini.union(
            pynini.string_file(hindi_path("numbers/teens_and_ties.tsv")),
            pynini.string_file(hindi_path("numbers/teens_and_ties_en.tsv")),
        ).optimize()
        teens_and_ties = add_weight(teens_ties, -0.1)
        self.digit = digit.optimize()
        self.zero = zero.optimize()
        self.teens_ties = teens_ties
        self.teens_and_ties = teens_and_ties.optimize()
        single_digit = digit | zero
        self.single_digits_graph = (single_digit + (insert(" ") + single_digit).star).optimize()

        drop_zero = add_weight(delete(ALL_ZERO), -0.1)

        def zeros(count):
            # ``fst ** 0`` is a Kleene star in pynini, not the empty string.
            return pynini.accep("") if count == 0 else drop_zero**count

        def suffixed(head, suffix, count):
            return head + insert(suffix) + zeros(count)

        def larger(head, suffix, count, tail):
            return head + insert(suffix) + zeros(count) + insert(" ") + tail

        hundreds = suffixed(digit, " सौ", 2)
        hundreds |= larger(digit, " सौ", 1, digit)
        hundreds |= larger(digit, " सौ", 0, teens_ties)
        self.graph_hundreds = hundreds.optimize()

        as_thousand = suffixed(teens_and_ties, " सौ", 2)
        as_thousand |= larger(teens_and_ties, " सौ", 1, digit)
        as_thousand |= larger(teens_and_ties, " सौ", 0, teens_ties)
        self.graph_hundreds_as_thousand = as_thousand.optimize()

        # tails[k] reads a k-digit remainder that has at least one non-zero digit.
        tails = {1: digit, 2: teens_ties, 3: self.graph_hundreds}
        levels = []
        for index, scale in enumerate(SCALES):
            low = index * 2 + 3
            graphs = []
            for head, width in ((digit, low + 1), (teens_and_ties, low + 2)):
                graph = suffixed(head, " " + scale, low)
                for size in range(1, low + 1):
                    graph |= larger(head, " " + scale, low - size, tails[size])
                graphs.append(graph.optimize())
            tails[low + 1], tails[low + 2] = graphs
            levels.extend(graphs)
        self.graph_thousands = tails[4]
        self.graph_ten_thousands = tails[5]
        self.graph_lakhs = tails[6]
        self.graph_ten_lakhs = tails[7]
        self.graph_without_leading_zeros = pynini.union(
            digit, zero, teens_and_ties, self.graph_hundreds, *levels).optimize()

        leading_zeros = (ALL_ZERO + ALL_DIGIT.star) @ self.single_digits_graph
        leading_zeros = add_weight(leading_zeros, 0.5)

        separator = delete(",")
        two = ALL_DIGIT**2
        three = ALL_DIGIT**3
        indian = pynini.closure(ALL_DIGIT, 1, 2) + (separator + two).star + separator + three
        western = pynini.closure(ALL_DIGIT, 1, 3) + (separator + three).plus
        separated = ((indian | western).optimize() @ self.graph_without_leading_zeros).optimize()

        self.final_graph = (self.graph_without_leading_zeros | leading_zeros | separated).optimize()

        # Address and serial codes: 1-3 digit groups are cardinals, 4+ digits are read digit by digit.
        limited = (digit | zero | self.teens_and_ties | self.graph_hundreds).optimize()
        digitwise = ((ALL_DIGIT**4 + ALL_DIGIT.star) @ self.single_digits_graph).optimize()
        self.code_num_graph = (limited | digitwise).optimize()
        self.limited_graph = (limited | self.graph_thousands | self.graph_ten_thousands).optimize()
