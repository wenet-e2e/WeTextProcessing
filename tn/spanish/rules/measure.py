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

from tn.processor import Processor
from tn.spanish.rules.utils import (
    ALPHA,
    ONES,
    SIGMA,
    data_file,
    data_labels,
    restrict,
    rewrite,
    strip_cardinal_apocope,
)

TOKEN_ORDER = {
    "measure": [
        "prefix", "negative", "integer", "integer_part", "fractional_part", "quantity",
        "numerator", "denominator", "value", "units", "math",
    ],
}


def _units():
    """Unit symbols to singular/plural names, split by grammatical gender."""
    plurals, genders = {}, {}
    for gender in ("fem", "masc"):
        for singular, plural in data_labels("measures/measurements_plural_" + gender + ".tsv"):
            plurals[singular] = plural
            genders[singular] = gender
    tables = {}
    for kind, name in (("simple", "measurements"), ("complex", "measurements_complex")):
        for symbol, singular in data_labels("measures/" + name + ".tsv"):
            if singular in plurals:
                gender = genders[singular]
                tables.setdefault((kind, gender, "singular"), []).append((symbol, singular))
                tables.setdefault((kind, gender, "plural"), []).append((symbol, plurals[singular]))
    return {key: pynini.string_map(rows).optimize() for key, rows in tables.items()}


class Measure(Processor):
    """Raw numbers and unit symbols; gender and number agreement happen in V."""

    def __init__(self, cardinal, decimal, fraction, deterministic=True):
        super().__init__("measure", token_orders=TOKEN_ORDER)
        self.deterministic = deterministic
        tables = _units()

        def table(kind, gender, number):
            return tables.get((kind, gender, number), pynini.Fst())

        all_singular = table("simple", "fem", "singular") | table("simple", "masc", "singular")
        denominator = (pynini.cross("/", "por ") + all_singular).optimize()
        optional_denominator = pynini.closure(pynutil.insert(" ") + denominator, 0, 1)
        units = {}
        for gender in ("fem", "masc"):
            singular = table("simple", gender, "singular")
            plural = table("simple", gender, "plural")
            units[gender, "singular"] = singular + optional_denominator | table("complex", gender, "singular")
            units[gender, "plural"] = plural + optional_denominator | table("complex", gender, "plural")
            # Fractions take a singular unit ("tres quintos de metro").
            units[gender, "fraction"] = units[gender, "singular"] | pynutil.add_weight(plural, 0.0001)
        for number in ("singular", "plural", "fraction"):
            units["masc", number] |= denominator
        units = {key: graph.optimize() for key, graph in units.items()}
        singular_units = units["fem", "singular"] | units["masc", "singular"]
        plural_units = units["fem", "plural"] | units["masc", "plural"]
        fraction_units = units["fem", "fraction"] | units["masc", "fraction"]

        # Tagger: the written number fields plus the written unit symbol.
        sign_t = cardinal.sign_tagger
        separator = pynini.closure(pynutil.delete(" "), 0, 1) + pynutil.insert(" ")
        one = pynini.accep("1") @ cardinal.graph
        other = restrict(cardinal.graph, pynini.accep("1"))
        standard_t = (
            sign_t + self.tag_field("integer", one) + separator + self.tag_field("units", singular_units)
            | sign_t + self.tag_field("integer", other) + separator + self.tag_field("units", plural_units)
            | decimal.fields_tagger + separator + self.tag_field("units", plural_units)
            | fraction.fields_tagger + separator + self.tag_field("units", fraction_units)
        )
        alpha = pynini.closure(ALPHA, 1)
        alpha_unit = pynini.accep("-") + alpha
        times = pynini.union("x", "X")
        integer_t = self.tag_field("integer", cardinal.graph)
        alpha_t = (
            (integer_t | decimal.unsigned_tagger) + pynutil.insert(" ") + self.tag_field("units", alpha_unit | times)
            | self.tag_field("prefix", alpha + "-") + pynutil.insert(" ") + (integer_t | decimal.unsigned_tagger)
        )

        delimiter = pynini.accep(" ") | pynutil.insert(" ")
        symbols = data_file("measures/math_symbols.tsv")
        numbers = cardinal.graph | pynutil.add_weight(cardinal.graph @ pynini.cross("un", "uno"), -1) | ALPHA
        equals = pynini.cross("=", "es igual a")
        operations = pynini.closure(delimiter + symbols + delimiter + numbers, 1)
        self.math_graph = (
            numbers + operations + delimiter + equals + delimiter + numbers
            | numbers + delimiter + equals + numbers + operations
        ).optimize()
        self.tagger = self.add_tokens(standard_t | alpha_t | self.tag_field("math", self.math_graph))

        # Verbalizer: numbers agree with the unit that follows them.
        sign_v = cardinal.sign_verbalizer
        to_uno = pynini.cross(ONES, "uno")

        def before_por(graph):
            return rewrite(graph, to_uno, right="[EOS]")

        def unit_v(gender, number):
            graph = units[gender, number]
            por = (graph @ ("por" + SIGMA)).optimize()
            other_units = (graph @ pynini.difference(SIGMA, "por" + SIGMA)).optimize()
            por_v = None if por.num_states() == 0 else self.verbalize_field("units", por, leading_space=True)
            return self.verbalize_field("units", other_units, leading_space=True), por_v

        space = pynutil.insert(" ")
        branches = []
        for gender in ("fem", "masc"):
            spoken = cardinal.fem if gender == "fem" else cardinal.masc
            card_one = sign_v + self.verbalize_field("integer", pynini.accep("1") @ spoken)
            card_other = sign_v + self.verbalize_field("integer", restrict(spoken, pynini.accep("1")))
            if gender == "fem":
                plural_numbers = card_other | decimal.sign_verbalizer + decimal.fem_verbalizer
                fraction_numbers = fraction.fem_verbalizer
                medio = pynini.union("media", "medias")
            else:
                plural_numbers = card_other | decimal.sign_verbalizer + decimal.masc_verbalizer
                fraction_numbers = fraction.masc_verbalizer
                medio = pynini.union("medio", "medios")
            for numbers_v, number in ((card_one, "singular"), (plural_numbers, "plural")):
                other_units, por_units = unit_v(gender, number)
                branches.append(numbers_v + space + other_units)
                if por_units is not None:
                    branches.append(before_por(numbers_v) + space + por_units)
            other_units, por_units = unit_v(gender, "fraction")
            fraction_units_v = other_units if por_units is None else other_units | por_units
            branches.append(fraction_numbers + pynutil.insert(" de ") + fraction_units_v)
            # "medio litro", not "medio de litro".
            ends_medio = (fraction_numbers @ (SIGMA + medio)).optimize()
            branches.append(pynutil.add_weight(ends_medio + space + fraction_units_v, -0.001))
        # Quantities are masculine nouns: "dos millones de gramos".
        quantity = decimal.sign_verbalizer + decimal.quantity_verbalizer + pynutil.insert(" de ")
        other_units, por_units = unit_v("masc", "plural")
        branches.append(quantity + other_units)
        if por_units is not None:
            branches.append(quantity + por_units)

        # Alphanumeric forms ("1,2-a", "a-5", "5x").
        alpha_numbers = (
            self.verbalize_field("integer", strip_cardinal_apocope(cardinal.graph))
            | decimal.masc_verbalizer | decimal.quantity_verbalizer
        )
        alpha_v = alpha_numbers + space + self.verbalize_field(
            "units", pynutil.delete("-") + alpha | times, leading_space=True,
        )
        prefix_numbers = (
            self.verbalize_field("integer", cardinal.graph, leading_space=True)
            | pynutil.delete(" ") + (decimal.masc_verbalizer | decimal.quantity_verbalizer)
        )
        alpha_v |= self.verbalize_field("prefix", alpha + pynutil.delete("-")) + space + prefix_numbers
        branches.append(pynutil.add_weight(alpha_v, 0.01))
        branches.append(self.verbalize_field("math", self.math_graph))
        self.verbalizer = self.delete_tokens(pynini.union(*branches))
