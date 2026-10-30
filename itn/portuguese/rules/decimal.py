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
from pynini.lib.pynutil import delete, insert

from itn.portuguese.rules.cardinal import SCALES
from tn.processor import Processor


TOKEN_ORDER = {}


class Decimal(Processor):
    """Raw spoken decimals; "vírgula" writes a comma and "ponto" a period.

    The fractional part is any sequence of cardinals below one thousand,
    e.g. "três vírgula quatorze quinze" -> 3,1415. Large round numbers keep
    their quantity word: "dois milhões" -> 2 milhões.
    """

    def __init__(self, cardinal):
        super().__init__("decimal", ordertype="itn", token_orders=TOKEN_ORDER)
        space = self.DELETE_SPACE
        self.fractional_graph = (
            (cardinal.numbers_up_to_thousand + space).star + cardinal.numbers_up_to_thousand
        ).optimize()
        self.point_graph = pynini.cross("vírgula", ",") | pynini.cross("ponto", ".")
        self.integer_graph = (cardinal.graph_no_exception | cardinal.zero).optimize()
        self.quantity_graph = pynini.union(*[word for scale in SCALES for word in scale])
        strip = delete("0").star + (self.DIGIT - "0") + self.DIGIT.star
        self.quantity_integer = (cardinal.numbers_up_to_million @ strip).optimize()
        self.negative = cardinal.negative

        def field_t(name, graph):
            return self.tag_field(name, graph)

        gap = delete(self.SPACE.plus) + insert(" ")
        decimal_t = (
            (field_t("integer_part", self.integer_graph) + gap).ques
            + field_t("point", self.point_graph) + gap + field_t("fractional_part", self.fractional_graph)
        )
        quantity_t = gap + field_t("quantity", self.quantity_graph)
        self.unsigned_tagger = (
            decimal_t + quantity_t.ques | field_t("integer_part", self.quantity_integer) + quantity_t
        ).optimize()
        sign_t = (field_t("negative", self.negative) + gap).ques
        self.fields_tagger = (sign_t + self.unsigned_tagger).optimize()

        integer_v = self.verbalize_field("integer_part", self.integer_graph)
        decimal_v = (
            (integer_v + delete(" ")).ques
            + self.verbalize_field("point", self.point_graph)
            + self.verbalize_field("fractional_part", self.fractional_graph, leading_space=True)
        )
        quantity_v = insert(" ") + self.verbalize_field("quantity", self.quantity_graph, leading_space=True)
        self.unsigned_verbalizer = (
            decimal_v + quantity_v.ques
            | self.verbalize_field("integer_part", self.quantity_integer) + quantity_v
        ).optimize()
        sign_v = (self.verbalize_field("negative", self.negative) + delete(" ")).ques
        self.fields_verbalizer = (sign_v + self.unsigned_verbalizer).optimize()
        self.tagger = self.add_tokens(self.fields_tagger)
        self.verbalizer = self.delete_tokens(self.fields_verbalizer)
