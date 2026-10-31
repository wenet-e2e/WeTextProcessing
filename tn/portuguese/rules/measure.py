# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from tn.processor import Processor
from tn.utils import get_abs_path


TOKEN_ORDER = {}


class Measure(Processor):
    """Flat raw measures: a cardinal, decimal or fraction followed by a unit symbol.

    A cardinal 1 takes the singular unit; hours make the cardinal feminine
    (1 h -> uma hora, 2 h -> duas horas).
    """

    def __init__(self, cardinal, decimal, fraction):
        super().__init__("measure", token_orders=TOKEN_ORDER)
        singular = pynini.string_file(get_abs_path("portuguese/data/measure/measurements_singular.tsv"))
        plural = pynini.string_file(get_abs_path("portuguese/data/measure/measurements_plural.tsv"))
        hours = pynini.union("hora", "horas")
        not_hours = pynini.difference(self.VSIGMA, hours)
        singular_hours = (singular @ hours).optimize()
        plural_hours = (plural @ hours).optimize()
        singular_other = (singular @ not_hours).optimize()
        plural_other = (plural @ not_hours).optimize()
        units = (singular | plural).optimize()

        sign_t = (self.tag_field("negative", cardinal.sign) + insert(" ")).ques
        sign_v = (self.verbalize_field("negative", cardinal.sign) + delete(" ")).ques
        number_t = (
            sign_t + self.tag_field("integer", cardinal.graph)
            | decimal.fields_tagger
            | fraction.fields_tagger
        )
        self.tagger = self.add_tokens(number_t + self.DELETE_SPACE + insert(" ") + self.tag_field("units", units))

        def unit(graph):
            return insert(" ") + self.verbalize_field("units", graph, leading_space=True)

        one = pynini.accep("1")
        other = pynini.difference(self.VSIGMA, "1")
        cardinal_v = (
            sign_v + self.verbalize_field("integer", one @ cardinal.graph_fem) + unit(singular_hours)
            | sign_v + self.verbalize_field("integer", other @ cardinal.graph_fem) + unit(plural_hours)
            | sign_v + self.verbalize_field("integer", one @ cardinal.graph) + unit(singular_other)
            | sign_v + self.verbalize_field("integer", other @ cardinal.graph) + unit(plural_other)
        )
        number_v = (decimal.fields_verbalizer | fraction.fields_verbalizer) + unit(plural)
        self.verbalizer = self.delete_tokens(cardinal_v | number_v)
