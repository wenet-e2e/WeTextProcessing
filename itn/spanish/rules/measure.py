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
from pynini.lib import pynutil

from itn.spanish.rules.utils import ALPHA, DELETE_SPACE, cased, case_insensitive, data_file, restrict
from tn.processor import Processor

TOKEN_ORDER = {
    "measure": [
        "negative", "integer", "integer_part", "point", "fractional_part", "quantity",
        "numerator", "denominator", "units", "math",
    ],
}


class Measure(Processor):
    """Raw spoken amounts and unit names; V writes numbers and unit symbols."""

    def __init__(self, cardinal, decimal, fraction, cased_input=False):
        super().__init__("measure", ordertype="itn", token_orders=TOKEN_ORDER)
        singular = case_insensitive(pynini.invert(data_file("measures/measurements_singular.tsv")))
        plural = case_insensitive(pynini.invert(data_file("measures/measurements_plural.tsv")))
        per = pynutil.insert("/") + pynutil.delete("por") + DELETE_SPACE + singular

        def units(graph):
            return (graph | per | pynutil.add_weight(graph + DELETE_SPACE + per, 0.01)).optimize()

        singular_units, plural_units = units(singular), units(plural)
        self.units_graph = (singular_units | plural_units).optimize()

        # Tagger.
        space = pynini.accep(" ")
        sign_t = cardinal.sign_tagger
        ones = cased(pynini.cross(pynini.union("un", "una", "uno"), "1"), cardinal.cased_input)
        others = restrict(cardinal.graph_no_exception, ones)
        one_t = sign_t + self.tag_field("integer", ones)
        other_t = sign_t + self.tag_field("integer", others)
        singular_t = space + self.tag_field("units", singular_units)
        plural_t = space + self.tag_field("units", plural_units)
        of = pynini.closure(pynutil.delete(cased(pynini.accep("de"), cased_input)) + DELETE_SPACE, 0, 1)
        conjunction = pynutil.delete(cased(pynini.union(" y", " con"), cased_input))
        graph = sign_t + decimal.unsigned_tagger + plural_t
        cardinal_t = other_t + plural_t | one_t + singular_t
        graph |= cardinal_t
        # "tres quintos de metro", "medio litro", "dos metros y medio".
        proper = sign_t + fraction.proper_tagger
        graph |= proper + space + (of + self.tag_field("units", singular_units) | self.tag_field("units", plural_units))
        graph |= cardinal_t + conjunction + space + fraction.proper_tagger
        graph |= pynutil.add_weight(fraction.mixed_tagger + plural_t, 0.001)

        symbols = cased(data_file("measures/math_symbols.tsv"), cased_input)
        equals = cased(pynini.string_map([("es igual a", "="), ("igual a", "=")]), cased_input)
        numbers = cardinal.graph_no_exception
        long_side = numbers + space + pynini.closure(symbols + space + numbers + pynini.closure(space, 0, 1), 1)
        short_side = numbers | ALPHA
        self.math_graph = (
            long_side + equals + space + short_side | short_side + space + equals + space + long_side
        ).optimize()
        graph |= self.tag_field("math", self.math_graph)
        self.tagger = self.add_tokens(graph)

        # Verbalizer.
        sign_v = cardinal.sign_verbalizer
        units_v = pynutil.insert(" ") + self.verbalize_field("units", self.units_graph, leading_space=True)
        integer_v = sign_v + self.verbalize_field("integer", ones | others)
        mixed_v = pynutil.delete(" ") + pynutil.insert(" ") + fraction.proper_verbalizer
        graph = (sign_v + decimal.unsigned_verbalizer | integer_v | fraction.fields_verbalizer) + units_v
        graph |= integer_v + mixed_v + units_v
        graph |= self.verbalize_field("math", self.math_graph)
        self.verbalizer = self.delete_tokens(graph)
