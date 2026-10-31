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
from pynini.lib.pynutil import add_weight, delete, insert

from itn.utils import get_abs_path
from tn.processor import Processor


TOKEN_ORDER = {}


class Measure(Processor):
    """Raw spoken quantities and unit names, e.g. "dois quilos" -> 2 kg."""

    def __init__(self, cardinal, decimal):
        super().__init__("measure", ordertype="itn", token_orders=TOKEN_ORDER)
        singular = pynini.string_file(get_abs_path("portuguese/data/measurements_singular.tsv")).invert()
        plural = pynini.string_file(get_abs_path("portuguese/data/measurements_plural.tsv")).invert()
        space = self.DELETE_SPACE
        gap = delete(self.SPACE.plus) + insert(" ")
        per = insert("/") + delete("por") + space + singular

        def units(graph):
            return (graph | per | add_weight(graph + space + per, 0.01)).optimize()

        singular_units = units(singular)
        plural_units = units(plural)
        self.units_graph = (singular_units | plural_units).optimize()
        one = pynini.cross("um", "1") | pynini.cross("uma", "1")
        other = pynini.difference(self.VSIGMA, pynini.union("um", "uma")) @ cardinal.graph_no_exception
        integer = (one | other).optimize()

        def field(name, graph):
            return self.tag_field(name, graph)

        sign_t = (field("negative", cardinal.negative) + gap).ques
        number_t = (
            field("integer", other) + gap + field("units", plural_units)
            | field("integer", one) + gap + field("units", singular_units)
            | decimal.unsigned_tagger + gap + field("units", plural_units)
        )
        self.tagger = self.add_tokens(sign_t + number_t)

        sign_v = (self.verbalize_field("negative", cardinal.negative) + delete(" ")).ques
        number_v = self.verbalize_field("integer", integer) | decimal.unsigned_verbalizer
        units_v = insert(" ") + self.verbalize_field("units", self.units_graph, leading_space=True)
        self.verbalizer = self.delete_tokens(sign_v + number_v + units_v)
