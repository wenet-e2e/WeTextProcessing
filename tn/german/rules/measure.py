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
from tn.utils import get_abs_path, load_labels


TOKEN_ORDER = {
    "measure": ["kind", "prefix", "negative", "integer", "integer_part", "fractional_part",
                "quantity", "numerator", "denominator", "units"],
}


class Measure(Processor):
    """Flat raw measures, including ordered alpha prefixes and unit suffixes."""

    def __init__(self, cardinal, decimal, fraction, deterministic=True):
        super().__init__("measure", token_orders=TOKEN_ORDER)
        self.deterministic = deterministic
        clean_space = pynini.cdrewrite(pynini.cross("\u00a0", " "), "", "", self.VSIGMA)
        labels = load_labels(get_abs_path("german/data/measure/measurements.tsv"))
        seen = {}
        ordered_labels = []
        for row in labels:
            symbol, word = row[:2]
            rank = seen.get(symbol, 0)
            seen[symbol] = rank + 1
            # Preserve resource weights and all candidates, using row order to
            # resolve equal-cost readings after converted source tags collapse.
            weight = (float(row[2]) if len(row) > 2 else 0.0) + rank * 0.00001
            ordered_labels.append((symbol, word, str(weight)))
        self.unit_singular_graph = (pynini.string_map(ordered_labels) @ clean_space).optimize()
        suppletive = pynini.string_file(get_abs_path("german/data/measure/suppletive.tsv")) @ clean_space
        self.unit_plural_graph = (
            self.unit_singular_graph @ pynini.cdrewrite(suppletive, "", "[EOS]", self.VSIGMA)
        ).optimize()
        self.unit_denominator_graph = (
            pynini.cross("/", "pro ") + self.unit_singular_graph
        ).optimize()
        optional_denominator = pynini.closure(pynutil.insert(" ") + self.unit_denominator_graph, 0, 1)
        self.units_singular_graph = (
            self.unit_singular_graph + optional_denominator | self.unit_denominator_graph
        ).optimize()
        self.units_plural_graph = (
            self.unit_plural_graph + optional_denominator | self.unit_denominator_graph
        ).optimize()
        self.units_graph = (self.units_singular_graph | self.units_plural_graph).optimize()

        sign_t = pynini.closure(self.tag_field("negative", cardinal.sign) + pynutil.insert(" "), 0, 1)
        sign_v = pynini.closure(self.verbalize_field("negative", cardinal.sign) + pynutil.delete(" "), 0, 1)
        change_one = pynini.cdrewrite(pynini.cross("eins", "ein"), "[BOS]", "[EOS]", self.VSIGMA)
        cardinal_one = (pynini.accep("1") @ cardinal.graph @ change_one).optimize()
        cardinal_other = ((pynini.closure(self.DIGIT, 1) - "1") @ cardinal.graph).optimize()
        one_t = sign_t + self.tag_field("integer", cardinal_one)
        other_t = sign_t + self.tag_field("integer", cardinal_other)
        one_v = sign_v + self.verbalize_field("integer", cardinal_one)
        other_v = sign_v + self.verbalize_field("integer", cardinal_other)
        unit_separator_t = pynutil.delete(pynini.closure(" ", 0, 1)) + pynutil.insert(" ")
        unit_separator_v = pynutil.insert(" ")
        standard_t = (
            one_t + unit_separator_t + self.tag_field("units", self.units_singular_graph)
            | (other_t | decimal.fields_tagger | fraction.fields_tagger)
            + unit_separator_t + self.tag_field("units", self.units_plural_graph)
        )
        standard_v = (
            one_v + unit_separator_v + self.verbalize_field("units", self.units_singular_graph, leading_space=True)
            | (other_v | decimal.fields_verbalizer | fraction.fields_verbalizer)
            + unit_separator_v + self.verbalize_field("units", self.units_plural_graph, leading_space=True)
        )

        # Dash/times paths are unsigned in the source; ordinary unit paths may
        # carry a negative sign. Prefix and suffix cannot share a schema field.
        unsigned_decimal_t = (
            pynini.difference(self.VSIGMA, "-" + self.VSIGMA) @ decimal.fields_tagger
        ).optimize()
        unsigned_decimal_v = (
            self.verbalize_field("integer_part", decimal.integer_graph) + (
                pynutil.insert(" ") + self.verbalize_field("quantity", decimal.quantity_graph, leading_space=True)
                | pynutil.insert(" komma ")
                + self.verbalize_field("fractional_part", decimal.fractional_graph, leading_space=True)
                + pynini.closure(
                    pynutil.insert(" ") + self.verbalize_field("quantity", decimal.quantity_graph, leading_space=True),
                    0, 1,
                )
            )
        ).optimize()
        bare_number_t = self.tag_field("integer", cardinal.graph) | unsigned_decimal_t
        bare_number_v = self.verbalize_field("integer", cardinal.graph) | unsigned_decimal_v
        alpha = pynini.closure(self.ALPHA, 1)
        suffix_t = bare_number_t + pynutil.delete("-") + pynutil.insert(" ") + self.tag_field("units", alpha)
        prefix_t = self.tag_field("prefix", alpha) + pynutil.delete("-") + pynutil.insert(" ") + bare_number_t
        suffix_v = bare_number_v + pynutil.insert(" ") + self.verbalize_field("units", alpha, leading_space=True)
        prefix_v = self.verbalize_field("prefix", alpha) + pynutil.delete(" ") + pynutil.insert(" ") + bare_number_v
        times_t = bare_number_t + pynutil.insert(" ") + self.tag_field("units", pynini.union("x", "X"))
        times_v = bare_number_v + pynutil.insert(" ") + self.verbalize_field("units", pynini.union("x", "X"), leading_space=True)

        self.tagger = self.add_tokens(
            pynutil.insert('kind: "unit" ') + standard_t
            | pynutil.insert('kind: "alpha" ') + (suffix_t | prefix_t)
            | pynutil.insert('kind: "times" ') + times_t
        )
        self.verbalizer = self.delete_tokens(
            pynutil.delete('kind: "unit" ') + standard_v
            | pynutil.delete('kind: "alpha" ') + (suffix_v | prefix_v)
            | pynutil.delete('kind: "times" ') + times_v
        )
