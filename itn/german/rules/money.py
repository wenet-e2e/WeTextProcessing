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
from tn.utils import get_abs_path


TOKEN_ORDER = {
    "money": ["currency", "integer_part", "fractional_part", "quantity", "conjunction",
              "minor_part", "currency_minor"],
}


class Money(Processor):

    def __init__(self, cardinal, decimal):
        super().__init__("money", ordertype="itn", token_orders=TOKEN_ORDER)
        whitespace = pynini.union(" ", "\t", "\n", "\r", "\u00a0")
        delete_space = pynutil.delete(whitespace.star)
        separator = pynutil.delete(whitespace.plus) + pynutil.insert(" ")
        change_one = pynini.cdrewrite(
            pynini.cross(pynini.union("ein", "eine"), "eins"), "[BOS]", "[EOS]", self.VSIGMA,
        )
        self.integer_graph = (change_one @ cardinal.graph_no_exception).optimize()
        self.currency_graph = pynini.string_file(get_abs_path("german/data/money/currency.tsv")).invert().optimize()
        minor = pynini.union(
            pynini.string_file(get_abs_path("german/data/money/currency_minor_singular.tsv")),
            pynini.string_file(get_abs_path("german/data/money/currency_minor_plural.tsv")),
        ).project("output").optimize()
        self.minor_graph = (
            self.integer_graph @ (self.DIGIT + self.DIGIT | pynutil.insert("0") + self.DIGIT)
        ).optimize()
        minor_word = pynutil.delete(minor)
        conjunction = pynutil.delete("und")
        cents_t = (
            self.tag_field("minor_part", self.minor_graph)
            + delete_space + pynutil.insert(" ") + self.tag_field("currency_minor", minor_word)
        )
        cents_v = (
            self.verbalize_field("minor_part", self.minor_graph)
            + self.verbalize_field("currency_minor", minor_word, leading_space=True)
        )
        optional_named_cents_t = pynini.closure(
            delete_space + pynutil.insert(" ")
            + pynini.closure(self.tag_field("conjunction", conjunction) + delete_space + pynutil.insert(" "), 0, 1)
            + cents_t, 0, 1,
        )
        optional_named_cents_v = pynini.closure(
            pynutil.delete(" ")
            + pynini.closure(self.verbalize_field("conjunction", conjunction) + pynutil.delete(" "), 0, 1)
            + pynutil.insert(",") + cents_v, 0, 1,
        )
        bare_minor = pynutil.add_weight(self.minor_graph, -0.7)
        optional_bare_cents_t = pynini.closure(separator + self.tag_field("minor_part", bare_minor), 0, 1)
        optional_bare_cents_v = pynini.closure(
            pynutil.insert(",") + self.verbalize_field("minor_part", bare_minor, leading_space=True), 0, 1,
        )
        unsigned = pynini.difference(self.VSIGMA, "minus " + self.VSIGMA)
        decimal_t = (unsigned @ decimal.fields_tagger).optimize()
        currency_t = self.tag_field("currency", self.currency_graph)
        integer_t = (
            self.tag_field("integer_part", self.integer_graph) + separator + currency_t
            + (optional_named_cents_t | optional_bare_cents_t)
        )
        integer_v = (
            self.verbalize_field("integer_part", self.integer_graph)
            + (optional_named_cents_v | optional_bare_cents_v)
        )
        self.tagger = self.add_tokens(
            integer_t | decimal_t + separator + currency_t | cents_t,
        )
        self.verbalizer = self.delete_tokens(
            self.verbalize_field("currency", self.currency_graph) + pynutil.delete(" ")
            + (integer_v | decimal.fields_verbalizer)
            | pynutil.insert("€0,") + cents_v,
        )
