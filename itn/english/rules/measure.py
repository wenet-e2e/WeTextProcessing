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
from pynini import closure, cross, invert, string_file
from pynini.lib.pynutil import add_weight, delete, insert

from itn.english.rules.cardinal import Cardinal
from itn.english.rules.decimal import Decimal
from tn.processor import Processor
from tn.utils import get_abs_path


class Measure(Processor):

    def __init__(self, cardinal=None, decimal=None):
        super().__init__(name="measure", ordertype="itn")
        self.cardinal = cardinal or Cardinal()
        self.decimal = decimal or Decimal(cardinal=self.cardinal)
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        ds = delete(" ")

        # Load measurements: symbol\tname, invert to get name -> symbol
        tsv_path = get_abs_path("../itn/english/data/measurements.tsv")
        units_graph = invert(string_file(tsv_path))

        # Handle plurals: generate plural->symbol mappings from the singular TSV entries
        # Uses finite string_map instead of cdrewrite to avoid slow runtime compose
        singular_names = {}
        with open(tsv_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t", 1)
                if len(parts) == 2:
                    singular_names.setdefault(parts[1], parts[0])

        plural_pairs = []
        irregular_plurals = {
            "foot": "feet", "inch": "inches",
            "ounce": "ounces",
        }
        for name, symbol in singular_names.items():
            if name in irregular_plurals:
                plural_pairs.append((irregular_plurals[name], symbol))
            elif name.endswith(("s", "z", "sh", "ch", "x")):
                plural_pairs.append((name + "es", symbol))
            elif name.endswith("y") and len(name) > 1 and name[-2] not in "aeiou":
                plural_pairs.append((name[:-1] + "ies", symbol))
            else:
                plural_pairs.append((name + "s", symbol))

        unit_singular = units_graph
        unit_plural = pynini.string_map(plural_pairs)

        unit = unit_singular | unit_plural

        # Handle "per" units: "per hour" -> "/h"
        per_unit = add_weight(insert("/") + delete("per") + ds + unit_singular, 1)
        full_unit = unit + closure(ds + per_unit, 0, 1) | per_unit

        # Cardinal value
        cardinal_value = self.cardinal.graph

        # Decimal value (reuse decimal's internal graph for the number)
        decimal_digit = string_file(get_abs_path("../itn/english/data/numbers/digit.tsv"))
        decimal_zero = string_file(get_abs_path("../itn/english/data/numbers/zero.tsv"))
        frac_digit = decimal_digit | decimal_zero | cross("o", "0")
        frac_graph = closure(frac_digit + ds) + frac_digit
        decimal_value = cardinal_value + ds + delete("point") + ds + insert(".") + frac_graph

        # Optional minus/negative prefix
        minus = delete("minus") | delete("negative")
        optional_sign = closure(insert("-") + minus + ds, 0, 1)

        # "point X" with no integer part
        point_only = delete("point") + ds + insert(".") + frac_graph

        number = optional_sign + (decimal_value | cardinal_value | point_only)

        value = insert('value: "') + number + insert('"')
        units = insert('units: "') + full_unit + insert('"')

        final_graph = value + ds + insert(" ") + units
        self.tagger = self.add_tokens(final_graph)

    def build_verbalizer(self):
        value = (
            delete("value:")
            + self.DELETE_SPACE
            + delete('"')
            + self.NOT_QUOTE.plus
            + delete('"')
        )
        units = (
            delete("units:")
            + self.DELETE_SPACE
            + delete('"')
            + self.NOT_QUOTE.plus
            + delete('"')
        )
        graph = value + self.DELETE_SPACE + insert(" ") + units
        self.verbalizer = self.delete_tokens(graph)
