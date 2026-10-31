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

from tn.italian.rules.utils import load_pairs, ranked_map
from tn.processor import Processor


def pluralize(name, suppletive):
    """Applies the longest suppletive suffix, e.g. ``centimetro`` -> ``centimetri``.

    As in NeMo, units without a suppletive entry (``ampere``, ``percento``)
    stay invariant.
    """

    for singular in sorted(suppletive, key=len, reverse=True):
        if name.endswith(singular):
            return name[:-len(singular)] + suppletive[singular]
    return name


class Measure(Processor):
    """Quantities with units, e.g. ``5 km/s`` -> ``cinque chilometri per secondo``.

    The tagger keeps the written number fields and the unit symbol (with an
    optional ``/unit`` denominator); the verbalizer reads the number and
    names the unit, singular after ``1`` and plural otherwise.
    """

    def __init__(self, cardinal, decimal):
        super().__init__("measure", token_orders={})
        units = load_pairs("italian/data/measure/measurements.tsv")
        suppletive = dict(load_pairs("italian/data/measure/suppletive.tsv"))
        singular = ranked_map(units)
        plural = ranked_map([(key, pluralize(name, suppletive)) for key, name in units])
        denominator = pynini.cross("/", " per ") + singular
        self.singular_units = (singular + denominator.ques).optimize()
        self.plural_units = (plural + denominator.ques).optimize()
        units_graph = self.singular_units | self.plural_units

        sign = cardinal.sign
        sign_tag = (self.tag_field("negative", sign) + insert(" ")).ques
        sign_verb = (self.verbalize_field("negative", sign) + delete(" ") + insert(" ")).ques
        integer_tag = self.tag_field("integer", cardinal.graph)
        decimal_tag = self.tag_field("integer_part", decimal.integer) + delete(",") + insert(" ")
        decimal_tag += self.tag_field("fractional_part", decimal.fractional)
        units_tag = delete(" ").ques + insert(" ") + self.tag_field("units", units_graph)
        self.tagger = self.add_tokens(sign_tag + (integer_tag | decimal_tag) + units_tag)

        one = self.verbalize_field("integer", pynini.cross("1", "un"))
        many = self.verbalize_field("integer", (self.VSIGMA - "1") @ cardinal.graph)
        fractional = self.verbalize_field("integer_part", decimal.integer) + insert(" virgola")
        fractional += insert(" ") + self.verbalize_field("fractional_part", decimal.fractional, leading_space=True)

        def units_verb(graph):
            return insert(" ") + self.verbalize_field("units", graph, leading_space=True)

        verbalizer = one + units_verb(self.singular_units) | (many | fractional) + units_verb(self.plural_units)
        self.verbalizer = self.delete_tokens(sign_verb + verbalizer)
