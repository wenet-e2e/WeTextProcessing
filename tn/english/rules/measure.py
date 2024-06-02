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

from tn.processor import Processor
from tn.utils import get_abs_path, load_labels, get_formats
from tn.english.rules.cardinal import Cardinal
from tn.english.rules.ordinal import Ordinal
from tn.english.rules.decimal import Decimal
from tn.english.rules.fraction import Fraction

suppletive = pynini.string_file(get_abs_path("english/data/suppletive.tsv"))
# _v = pynini.union("a", "e", "i", "o", "u")
_c = pynini.union("b", "c", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p",
                  "q", "r", "s", "t", "v", "w", "x", "y", "z")
_ies = pynini.closure(Processor("tmp").VCHAR) + _c + pynini.cross("y", "ies")
_es = pynini.closure(Processor("tmp").VCHAR) + pynini.union(
    "s", "sh", "ch", "x", "z") + pynutil.insert("es")
_s = pynini.closure(Processor("tmp").VCHAR) + pynutil.insert("s")

graph_plural = plurals._priority_union(
    suppletive,
    plurals._priority_union(
        _ies,
        plurals._priority_union(_es, _s,
                                pynini.closure(Processor("tmp").VCHAR)),
        pynini.closure(Processor("tmp").VCHAR)),
    pynini.closure(Processor("tmp").VCHAR)).optimize()
SINGULAR_TO_PLURAL = graph_plural


class Measure(Processor):

    def __init__(self, deterministic: bool = False):
        """
        Args:
            deterministic: if True will provide a single transduction option,
                for False multiple transduction are generated (used for audio-based normalization)
        """
        super().__init__('measure', ordertype="en_tn")
        self.deterministic = deterministic
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        """
        Finite state transducer for classifying measure, suppletive aware, e.g.
            -12kg -> measure { negative: "true" integer: "twelve" units: "kilograms" }
            1kg -> measure { integer: "one" units: "kilogram" }
            .5kg -> measure { fractional_part: "five" units: "kilograms" }
        """
        cardinal = Cardinal(self.deterministic)
        cardinal_graph = cardinal.graph_with_and | self.get_range(
            cardinal.graph_with_and)

        graph_unit = pynini.string_file(
            get_abs_path("english/data/measure/unit.tsv"))
        if not self.deterministic:
            graph_unit |= pynini.string_file(
                get_abs_path("english/data/measure/unit_alternatives.tsv"))

        graph_unit |= pynini.compose(
            pynini.closure(self.TO_LOWER, 1) + (self.ALPHA | self.TO_LOWER) +
            pynini.closure(self.ALPHA | self.TO_LOWER), graph_unit).optimize()

        graph_unit_plural = graph_unit @ SINGULAR_TO_PLURAL

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0,
            1)

        graph_unit2 = (pynini.cross("/", "per") +
                       self.DELETE_ZERO_OR_ONE_SPACE + pynutil.insert(" ") +
                       graph_unit)

        optional_graph_unit2 = pynini.closure(
            self.DELETE_ZERO_OR_ONE_SPACE + pynutil.insert(" ") + graph_unit2,
            0,
            1,
        )

        unit_plural = (
            pynutil.insert(" units: \"") +
            (graph_unit_plural + optional_graph_unit2 | graph_unit2) +
            pynutil.insert("\""))

        unit_singular = (pynutil.insert(" units: \"") +
                         (graph_unit + optional_graph_unit2 | graph_unit2) +
                         pynutil.insert("\""))

        decimal = Decimal(self.deterministic)
        subgraph_decimal = (optional_graph_negative +
                            decimal.final_graph_wo_negative +
                            pynini.accep(' ').ques + unit_plural)

        # support radio FM/AM
        subgraph_decimal |= (decimal.final_graph_wo_negative +
                             pynini.accep(' ').ques +
                             pynutil.insert(" units: \"") +
                             pynini.union("AM", "FM") + pynutil.insert("\""))

        subgraph_cardinal = (
            optional_graph_negative + pynutil.insert("integer: \"") +
            ((pynini.closure(self.VCHAR) - "1") @ cardinal_graph) +
            pynutil.insert("\"") + pynini.accep(' ').ques + unit_plural)

        subgraph_cardinal |= (optional_graph_negative +
                              pynutil.insert("integer: \"") +
                              pynini.cross("1", "one") + pynutil.insert("\"") +
                              pynini.accep(' ').ques + unit_singular)

        unit_graph = (
            pynutil.insert("integer: \"-\" units: \"") +
            ((pynini.cross("/", "per") + self.DELETE_ZERO_OR_ONE_SPACE) |
             (pynini.accep("per") + pynutil.delete(" "))) +
            pynutil.insert(" ") + graph_unit +
            pynutil.insert("\" preserve_order: \"true\""))  # noqa

        decimal_dash_alpha = (decimal.final_graph_wo_negative +
                              pynini.cross('-', '') +
                              pynutil.insert(" units: \"") +
                              pynini.closure(self.ALPHA, 1) +
                              pynutil.insert("\""))

        decimal_times = (decimal.final_graph_wo_negative +
                         pynutil.insert(" units: \"") +
                         (pynini.cross(pynini.union('x', "X"), 'x')
                          | pynini.cross(pynini.union('x', "X"), ' times')) +
                         pynutil.insert("\""))

        alpha_dash_decimal = (pynutil.insert("units: \"") +
                              pynini.closure(self.ALPHA, 1) +
                              pynini.accep('-') + pynutil.insert("\"") +
                              decimal.final_graph_wo_negative +
                              pynutil.insert(" preserve_order: \"true\""))

        fraction = Fraction(self.deterministic)
        subgraph_fraction = (fraction.graph + pynini.accep(' ').ques +
                             unit_plural)

        address = self.get_address_graph(cardinal)
        address = (pynutil.insert("units: \"address\" integer: \"") + address +
                   pynutil.insert("\" preserve_order: \"true\""))

        math_operations = pynini.string_file(
            get_abs_path("english/data/measure/math_operation.tsv"))
        delimiter = pynini.accep(" ") | pynutil.insert(" ")

        math = ((cardinal_graph | self.ALPHA) + delimiter + math_operations +
                (delimiter | self.ALPHA) + cardinal_graph + delimiter +
                pynini.cross("=", "equals") + delimiter +
                (cardinal_graph | self.ALPHA))

        math |= ((cardinal_graph | self.ALPHA) + delimiter +
                 pynini.cross("=", "equals") + delimiter +
                 (cardinal_graph | self.ALPHA) + delimiter + math_operations +
                 delimiter + cardinal_graph)

        math = (pynutil.insert("units: \"math\" integer: \"") + math +
                pynutil.insert("\" preserve_order: \"true\""))
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
        range_graph = cardinal + pynini.cross(pynini.union("-", " - "),
                                              " to ") + cardinal

        for x in [" x ", "x"]:
            range_graph |= cardinal + pynini.cross(x, " by ") + cardinal
            if not self.deterministic:
                range_graph |= cardinal + pynini.cross(
                    x, " times ") + pynini.closure(cardinal, 0, 1)

        for x in ["*", " * "]:
            range_graph |= cardinal + pynini.cross(x, " times ") + cardinal
        return range_graph.optimize()

    def get_address_graph(self, cardinal: Processor):
        """
        Finite state transducer for classifying serial.
            The serial is a combination of digits, letters and dashes, e.g.:
            2788 San Tomas Expy, Santa Clara, CA 95051 ->
                units: "address" integer: "two seven eight eight San Tomas Expressway Santa Clara California nine five zero five one" preserve_order: \"true\"
        """
        ordinal = Ordinal(self.deterministic)
        ordinal_verbalizer = ordinal.graph_v
        ordinal_tagger = ordinal.graph
        ordinal_num = pynini.compose(
            pynutil.insert("integer: \"") + ordinal_tagger +
            pynutil.insert("\""), ordinal_verbalizer)

        address_num = self.DIGIT**(
            1,
            2) @ cardinal.graph_hundred_component_at_least_one_none_zero_digit
        address_num += self.INSERT_SPACE + self.DIGIT**2 @ (
            pynini.closure(pynini.cross("0", "zero "), 0, 1) +
            cardinal.graph_hundred_component_at_least_one_none_zero_digit)
        # to handle the rest of the numbers
        address_num = pynini.compose(self.DIGIT**(3, 4), address_num)
        address_num = plurals._priority_union(address_num, cardinal.graph,
                                              pynini.closure(self.VCHAR))

        direction = (pynini.cross("E", "East")
                     | pynini.cross("S", "South")
                     | pynini.cross("W", "West")
                     | pynini.cross("N", "North")) + pynini.closure(
                         pynutil.delete("."), 0, 1)

        direction = pynini.closure(pynini.accep(" ") + direction, 0, 1)
        address_words = get_formats(
            get_abs_path("english/data/address/address_word.tsv"))
        address_words = (
            pynini.accep(" ") +
            (pynini.closure(ordinal_num, 0, 1)
             | self.UPPER + pynini.closure(self.ALPHA, 1)) + " " +
            pynini.closure(self.UPPER + pynini.closure(self.ALPHA) + " ") +
            address_words)

        city = pynini.closure(self.ALPHA | pynini.accep(" "), 1)
        city = pynini.closure(
            pynini.accep(",") + pynini.accep(" ") + city, 0, 1)

        states = load_labels(get_abs_path("english/data/address/state.tsv"))

        additional_options = []
        for x, y in states:
            additional_options.append((x, f"{y[0]}.{y[1:]}"))
        states.extend(additional_options)
        state_graph = pynini.string_map(states)
        state = pynini.invert(state_graph)
        state = pynini.closure(
            pynini.accep(",") + pynini.accep(" ") + state, 0, 1)

        zip_code = pynini.compose(self.DIGIT**5, cardinal.single_digits_graph)
        zip_code = pynini.closure(
            pynini.closure(pynini.accep(","), 0, 1) + pynini.accep(" ") +
            zip_code,
            0,
            1,
        )

        address = address_num + direction + address_words + pynini.closure(
            city + state + zip_code, 0, 1)

        address |= address_num + direction + address_words + pynini.closure(
            pynini.cross(".", ""), 0, 1)

        return address

    def build_verbalizer(self):
        """
        Finite state transducer for verbalizing measure, e.g.
            measure { negative: "true" integer: "twelve" units: "kilograms" } -> minus twelve kilograms
            measure { integer_part: "twelve" fractional_part: "five" units: "kilograms" } -> twelve point five kilograms
        """
        cardinal = Cardinal(self.deterministic)
        unit = (pynutil.delete("units: \"") +
                pynini.difference(pynini.closure(self.NOT_QUOTE, 1),
                                  pynini.union("address", "math")) +
                pynutil.delete("\"") + self.DELETE_SPACE)

        if not self.deterministic:
            unit |= pynini.compose(
                unit, pynini.cross(pynini.union("inch", "inches"), "\""))

        decimal = Decimal(self.deterministic)
        graph_decimal = decimal.numbers

        if not self.deterministic:
            graph_decimal |= pynini.compose(
                graph_decimal,
                pynini.closure(self.VCHAR) +
                (pynini.cross(" point five", " and a half")
                 | pynini.cross("zero point five", "half")
                 | pynini.cross(" point two five", " and a quarter")
                 | pynini.cross("zero point two five", "quarter")),
            ).optimize()

        graph_cardinal = cardinal.numbers

        fraction = Fraction(self.deterministic)
        graph_fraction = fraction.graph_v

        graph = (graph_cardinal | graph_decimal
                 | graph_fraction) + pynini.accep(" ") + unit

        # SH adds "preserve_order: \"true\"" by default
        preserve_order = pynutil.delete(
            "preserve_order:") + self.DELETE_SPACE + pynutil.delete(
                "\"true\"") + self.DELETE_SPACE
        graph |= unit + self.INSERT_SPACE + (
            graph_cardinal |
            graph_decimal) + self.DELETE_SPACE + pynini.closure(preserve_order)
        # for only unit
        graph |= (pynutil.delete("integer: \"-\"") + self.DELETE_SPACE + unit +
                  pynini.closure(preserve_order))
        address = (pynutil.delete("units: \"address\" ") + self.DELETE_SPACE +
                   graph_cardinal + self.DELETE_SPACE +
                   pynini.closure(preserve_order))
        math = (pynutil.delete("units: \"math\" ") + self.DELETE_SPACE +
                graph_cardinal + self.DELETE_SPACE +
                pynini.closure(preserve_order))
        graph |= address | math

        delete_tokens = self.delete_tokens(graph)
        self.verbalizer = delete_tokens.optimize()
