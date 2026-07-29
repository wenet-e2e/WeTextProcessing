# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2024, WENET COMMUNITY.  Xingchen Song (sxc19@tsinghua.org.cn).
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
from pynini.examples import plurals
from pynini.lib import pynutil

from tn.english.rules.cardinal import Cardinal
from tn.english.rules.decimal import Decimal
from tn.english.rules.fraction import Fraction
from tn.english.rules.ordinal import Ordinal
from tn.processor import Processor
from tn.utils import get_abs_path, get_formats, load_labels

suppletive = pynini.string_file(get_abs_path("english/data/suppletive.tsv"))
# _v = pynini.union("a", "e", "i", "o", "u")
_c = pynini.union(
    "b",
    "c",
    "d",
    "f",
    "g",
    "h",
    "j",
    "k",
    "l",
    "m",
    "n",
    "p",
    "q",
    "r",
    "s",
    "t",
    "v",
    "w",
    "x",
    "y",
    "z",
)
_ies = Processor("tmp").VCHAR.star + _c + pynini.cross("y", "ies")
_es = Processor("tmp").VCHAR.star + pynini.union("s", "sh", "ch", "x", "z") + pynutil.insert("es")
_s = Processor("tmp").VCHAR.star + pynutil.insert("s")

graph_plural = plurals._priority_union(
    suppletive,
    plurals._priority_union(
        _ies,
        plurals._priority_union(_es, _s,
                                Processor("tmp").VCHAR.star),
        Processor("tmp").VCHAR.star,
    ),
    Processor("tmp").VCHAR.star,
).optimize()
SINGULAR_TO_PLURAL = graph_plural


class Measure(Processor):

    def __init__(self, deterministic: bool = False, cardinal=None, decimal=None, fraction=None, ordinal=None):
        super().__init__("measure", ordertype="en_tn")
        self.deterministic = deterministic
        self.cardinal = cardinal or Cardinal(deterministic)
        self.ordinal = ordinal or Ordinal(deterministic, cardinal=self.cardinal)
        self.decimal = decimal or Decimal(deterministic, cardinal=self.cardinal)
        self.fraction = fraction or Fraction(deterministic, cardinal=self.cardinal, ordinal=self.ordinal)
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        """
        Finite state transducer for classifying measure, suppletive aware, e.g.
            -12kg -> measure { negative: "-" integer: "12" units: "kg" }
            1kg -> measure { integer: "1" units: "kg" }
            .5kg -> measure { fractional_part: "5" units: "kg" }
        """
        cardinal = self.cardinal
        cardinal_graph = cardinal.graph_with_and | self.get_range(cardinal.graph_with_and)

        graph_unit = pynini.string_file(get_abs_path("english/data/measure/unit.tsv"))
        if not self.deterministic:
            graph_unit |= pynini.string_file(get_abs_path("english/data/measure/unit_alternatives.tsv"))

        graph_unit |= pynini.compose(
            self.TO_LOWER.plus + (self.ALPHA | self.TO_LOWER) + (self.ALPHA | self.TO_LOWER).star,
            graph_unit,
        ).optimize()

        graph_unit_plural = graph_unit @ SINGULAR_TO_PLURAL
        self.graph_unit = graph_unit
        self.graph_unit_plural = graph_unit_plural

        optional_graph_negative = (self.tag_field("negative", pynini.cross("-", "true")) + self.INSERT_SPACE).ques

        graph_unit2 = pynini.cross("/", "per") + self.DELETE_ZERO_OR_ONE_SPACE + pynutil.insert(" ") + graph_unit

        optional_graph_unit2 = (self.DELETE_ZERO_OR_ONE_SPACE + pynutil.insert(" ") + graph_unit2).ques

        self.unit_plural_graph = graph_unit_plural + optional_graph_unit2 | graph_unit2
        self.unit_singular_graph = graph_unit + optional_graph_unit2 | graph_unit2
        unit_plural = self.tag_field("units", self.unit_plural_graph)
        unit_singular = self.tag_field("units", self.unit_singular_graph)

        decimal = self.decimal
        subgraph_decimal = (optional_graph_negative + decimal.final_graph_wo_negative + pynutil.delete(" ").ques +
                            self.INSERT_SPACE + unit_plural)

        # support radio FM/AM
        subgraph_decimal |= (decimal.final_graph_wo_negative + pynutil.delete(" ").ques + self.INSERT_SPACE +
                             self.tag_field("units", pynini.union("AM", "FM")))

        self.cardinal_other_graph = (self.VCHAR.star - "1") @ cardinal_graph
        subgraph_cardinal = (optional_graph_negative + self.tag_field("integer", self.cardinal_other_graph) +
                             pynutil.delete(" ").ques + self.INSERT_SPACE + unit_plural)

        self.cardinal_one_graph = pynini.cross("1", "one")
        subgraph_cardinal |= (optional_graph_negative + self.tag_field("integer", self.cardinal_one_graph) +
                              pynutil.delete(" ").ques + self.INSERT_SPACE + unit_singular)

        self.unit_only_graph = ((pynini.cross("/", "per") + self.DELETE_ZERO_OR_ONE_SPACE) |
                                (pynini.accep("per") + pynutil.delete(" "))) + pynutil.insert(" ") + graph_unit
        unit_graph = (pynutil.insert('integer: "-" units: "') + self.input_projection(self.unit_only_graph) +
                      pynutil.insert('"'))  # noqa

        self.alpha_unit_graph = self.ALPHA.plus
        self.dash_separator_graph = pynutil.delete("-")
        decimal_dash_alpha = (decimal.final_graph_wo_negative + self.INSERT_SPACE +
                              self.tag_field("separator", self.dash_separator_graph) + self.INSERT_SPACE +
                              self.tag_field("units", self.alpha_unit_graph))

        self.times_unit_graph = pynini.cross(pynini.union("x", "X"), "x") | pynini.cross(pynini.union("x", "X"), " times")
        decimal_times = (decimal.final_graph_wo_negative + self.INSERT_SPACE + self.tag_field("units", self.times_unit_graph))

        self.alpha_dash_unit_graph = self.ALPHA.plus + pynini.accep("-")
        alpha_dash_decimal = (self.tag_field("units", self.alpha_dash_unit_graph) + self.INSERT_SPACE +
                              decimal.final_graph_wo_negative)

        fraction = self.fraction
        subgraph_fraction = (fraction.graph + pynutil.delete(" ").ques + self.INSERT_SPACE + unit_plural)

        self.address_graph = self.get_address_graph(cardinal)
        address = pynutil.insert('units: "address" ') + self.tag_field("integer", self.address_graph)

        math_operations = pynini.string_file(get_abs_path("english/data/measure/math_operation.tsv"))
        delimiter = pynini.accep(" ") | pynutil.insert(" ")

        math = ((cardinal_graph | self.ALPHA) + delimiter + math_operations + (delimiter | self.ALPHA) + cardinal_graph +
                delimiter + pynini.cross("=", "equals") + delimiter + (cardinal_graph | self.ALPHA))

        math |= ((cardinal_graph | self.ALPHA) + delimiter + pynini.cross("=", "equals") + delimiter +
                 (cardinal_graph | self.ALPHA) + delimiter + math_operations + delimiter + cardinal_graph)

        self.math_graph = math
        math = pynutil.insert('units: "math" ') + self.tag_field("integer", self.math_graph)
        final_graph = (subgraph_decimal
                       | subgraph_cardinal
                       | unit_graph
                       | decimal_dash_alpha
                       | decimal_times
                       | alpha_dash_decimal
                       | subgraph_fraction
                       | address
                       | math)

        final_graph = self.add_tokens(final_graph)
        self.tagger = final_graph.optimize()

    def get_range(self, cardinal: Processor):
        """
        Returns range forms for measure tagger, e.g. 2-3, 2x3, 2*2

        Args:
            cardinal: cardinal GraphFst
        """
        range_graph = cardinal + pynini.cross(pynini.union("-", " - "), " to ") + cardinal

        for x in [" x ", "x"]:
            range_graph |= cardinal + pynini.cross(x, " by ") + cardinal
            if not self.deterministic:
                range_graph |= cardinal + pynini.cross(x, " times ") + cardinal.ques

        for x in ["*", " * "]:
            range_graph |= cardinal + pynini.cross(x, " times ") + cardinal
        return range_graph.optimize()

    def get_address_graph(self, cardinal: Processor):
        """
        Finite state transducer for classifying serial.
            The serial is a combination of digits, letters and dashes, e.g.:
            2788 San Tomas Expy, Santa Clara, CA 95051 ->
                units: "address"
                integer: "2788 San Tomas Expy, Santa Clara, CA 95051"
        """
        ordinal = self.ordinal
        ordinal_verbalizer = ordinal.graph_v
        ordinal_tagger = ordinal.graph
        ordinal_num = pynini.compose(
            pynutil.insert('integer: "') + ordinal_tagger + pynutil.insert('"'),
            ordinal_verbalizer,
        )

        address_num = self.DIGIT**(1, 2) @ cardinal.graph_hundred_component_at_least_one_none_zero_digit
        address_num += self.INSERT_SPACE + self.DIGIT**2 @ (pynini.cross("0", "zero ").ques +
                                                            cardinal.graph_hundred_component_at_least_one_none_zero_digit)
        # to handle the rest of the numbers
        address_num = pynini.compose(self.DIGIT**(3, 4), address_num)
        address_num = plurals._priority_union(address_num, cardinal.graph, self.VCHAR.star)

        direction = (pynini.cross("E", "East")
                     | pynini.cross("S", "South")
                     | pynini.cross("W", "West")
                     | pynini.cross("N", "North")) + pynutil.delete(".").ques

        direction = (pynini.accep(" ") + direction).ques
        address_words = get_formats(get_abs_path("english/data/address/address_word.tsv"))
        address_words = (pynini.accep(" ") + (ordinal_num.ques | self.UPPER + self.ALPHA.plus) + " " +
                         (self.UPPER + self.ALPHA.star + " ").star + address_words)

        city = (self.ALPHA | pynini.accep(" ")).plus
        city = (pynini.accep(", ") + city).ques

        states = load_labels(get_abs_path("english/data/address/state.tsv"))

        additional_options = []
        for x, y in states:
            additional_options.append((x, f"{y[0]}.{y[1:]}"))
        states.extend(additional_options)
        state_graph = pynini.string_map(states)
        state = pynini.invert(state_graph)
        state = (pynini.accep(",") + pynini.accep(" ") + state).ques

        zip_code = pynini.compose(self.DIGIT**5, cardinal.single_digits_graph)
        zip_code = pynini.accep(",").ques + pynini.accep(" ") + zip_code

        address = address_num + direction + address_words + (city + state + zip_code).ques

        address |= address_num + direction + address_words + pynini.cross(".", "").ques

        return address

    def build_verbalizer(self):
        """
        Finite state transducer for verbalizing measure, e.g.
            measure { negative: "-" integer: "12" units: "kg" } -> minus twelve kilograms
            measure { integer_part: "12" fractional_part: "5" units: "kg" } -> twelve point five kilograms
        """
        cardinal = self.cardinal
        unit_plural = (pynutil.delete('units: "') + self.unit_plural_graph + pynutil.delete('"') + self.DELETE_SPACE)
        unit_singular = (pynutil.delete('units: "') + self.unit_singular_graph + pynutil.delete('"') + self.DELETE_SPACE)

        decimal = self.decimal
        graph_decimal = decimal.numbers

        if not self.deterministic:
            colloquial_fraction = pynini.compose(
                graph_decimal,
                self.VCHAR.star + (pynini.cross(" point five", " and a half")
                                   | pynini.cross("zero point five", "half")
                                   | pynini.cross(" point two five", " and a quarter")
                                   | pynini.cross("zero point two five", "quarter")),
            ).optimize()
            graph_decimal |= pynutil.add_weight(colloquial_fraction, 0.0001)

        fraction = self.fraction
        graph_fraction = fraction.graph_v

        optional_sign = cardinal.optional_sign
        integer_one = (pynutil.delete("integer:") + self.DELETE_SPACE + pynutil.delete('"') + self.cardinal_one_graph +
                       pynutil.delete('"'))
        integer_other = (pynutil.delete("integer:") + self.DELETE_SPACE + pynutil.delete('"') + self.cardinal_other_graph +
                         pynutil.delete('"'))
        graph_cardinal_one = optional_sign + integer_one
        graph_cardinal_other = optional_sign + integer_other

        graph = graph_cardinal_one + pynini.accep(" ") + unit_singular
        graph |= graph_cardinal_other + pynini.accep(" ") + unit_plural
        graph |= graph_decimal + pynini.accep(" ") + unit_plural
        graph |= graph_fraction + pynini.accep(" ") + unit_plural

        radio_unit = pynutil.delete('units: "') + pynini.union("AM", "FM") + pynutil.delete('"')
        graph |= graph_decimal + pynini.accep(" ") + radio_unit

        alpha_unit = pynutil.delete('units: "') + self.alpha_unit_graph + pynutil.delete('"') + self.DELETE_SPACE
        dash_separator = (self.DELETE_SPACE + pynutil.delete('separator: "') + self.dash_separator_graph +
                          pynutil.delete('"') + self.DELETE_SPACE)
        times_unit = pynutil.delete('units: "') + self.times_unit_graph + pynutil.delete('"') + self.DELETE_SPACE
        alpha_dash_unit = (pynutil.delete('units: "') + self.alpha_dash_unit_graph + pynutil.delete('"') + self.DELETE_SPACE)
        graph |= graph_decimal + dash_separator + self.INSERT_SPACE + alpha_unit
        graph |= graph_decimal + pynini.accep(" ") + times_unit
        graph |= alpha_dash_unit + self.INSERT_SPACE + graph_decimal + self.DELETE_SPACE

        # for only unit
        unit_only = pynutil.delete('units: "') + self.unit_only_graph + pynutil.delete('"') + self.DELETE_SPACE
        graph |= pynutil.delete('integer: "-"') + self.DELETE_SPACE + unit_only
        address = (pynutil.delete('units: "address" ') + self.DELETE_SPACE + pynutil.delete('integer: "') +
                   self.address_graph + pynutil.delete('"') + self.DELETE_SPACE)
        math = (pynutil.delete('units: "math" ') + self.DELETE_SPACE + pynutil.delete('integer: "') + self.math_graph +
                pynutil.delete('"') + self.DELETE_SPACE)
        graph |= address | math

        delete_tokens = self.delete_tokens(graph)
        self.verbalizer = delete_tokens.optimize()
