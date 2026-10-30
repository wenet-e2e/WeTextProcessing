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
from pynini.examples import plurals
from pynini.lib import pynutil

from tn.processor import Processor
from tn.utils import get_abs_path


TOKEN_ORDER = {
    "measure": ["prefix", "negative", "fraction_negative", "integer", "integer_part", "conjunction",
                "numerator", "denominator", "fractional_part", "quantity", "units"],
}


class Measure(Processor):

    def __init__(self, cardinal, decimal, fraction):
        super().__init__("measure", ordertype="itn", token_orders=TOKEN_ORDER)
        whitespace = pynini.union(" ", "\t", "\n", "\r", "\u00a0")
        delete_space = pynutil.delete(whitespace.star)
        separator = pynutil.delete(whitespace.plus) + pynutil.insert(" ")
        change_one = pynini.cdrewrite(
            pynini.cross(pynini.union("ein", "eine"), "eins"), "[BOS]", "[EOS]", self.VSIGMA,
        )
        self.integer_graph = (change_one @ cardinal.graph_no_exception).optimize()
        self.unit_singular_graph = pynini.string_file(
            get_abs_path("german/data/measure/measurements.tsv"),
        ).invert().optimize()
        suppletive = pynini.string_file(get_abs_path("german/data/measure/suppletive.tsv"))
        regular = pynini.union(
            self.VSIGMA + "e" + pynutil.insert("n"),
            self.VSIGMA + pynini.union(
                "ent", "and", "ant", "ist", "or", "ion", "ik", "heit", "keit", "schaft", "tät", "ung",
            ) + pynutil.insert("en"),
            self.VSIGMA + "in" + (pynutil.insert("e") | pynutil.insert("nen")),
            self.VSIGMA + pynini.union("ma", "um", "us") + pynutil.insert("en"),
            self.VSIGMA + pynini.union("eur", "ich", "ier", "ig", "ling", "ör") + pynutil.insert("e"),
            self.VSIGMA + pynini.union("a", "i", "o", "u", "y") + pynutil.insert("s"),
        )
        singular_to_plural = plurals._priority_union(suppletive, regular, self.VSIGMA).optimize()
        self.unit_plural_graph = (singular_to_plural.invert() @ self.unit_singular_graph).optimize()
        unit = (self.unit_singular_graph | self.unit_plural_graph).optimize()
        denominator = pynutil.insert("/") + pynutil.delete("pro") + delete_space + self.unit_singular_graph
        self.units_graph = (
            unit | denominator | pynutil.add_weight(unit + delete_space + denominator, 0.01)
        ).optimize()
        sign_t = pynini.closure(self.tag_field("negative", cardinal.sign) + separator, 0, 1)
        sign_v = pynini.closure(self.verbalize_field("negative", cardinal.sign) + pynutil.delete(" "), 0, 1)
        unsigned = pynini.difference(self.VSIGMA, "minus " + self.VSIGMA)
        decimal_t = (unsigned @ decimal.fields_tagger).optimize()
        fraction_t = (unsigned @ fraction.fields_tagger).optimize()
        fraction_sign_t = pynini.closure(
            self.tag_field("fraction_negative", cardinal.sign) + pynutil.delete(" ") + pynutil.insert(" "), 0, 1,
        )
        fraction_sign_v = pynini.closure(
            self.verbalize_field("fraction_negative", cardinal.sign) + pynutil.delete(" "), 0, 1,
        )
        number_t = (
            self.tag_field("integer", self.integer_graph) | decimal_t | fraction_sign_t + fraction_t
        )
        number_v = (
            self.verbalize_field("integer", self.integer_graph)
            | decimal.fields_verbalizer | fraction_sign_v + fraction.fields_verbalizer
        )
        self.tagger = self.add_tokens(sign_t + number_t + separator + self.tag_field("units", self.units_graph))
        self.verbalizer = self.delete_tokens(
            sign_v + number_v + pynutil.insert(" ")
            + self.verbalize_field("units", self.units_graph, leading_space=True),
        )
