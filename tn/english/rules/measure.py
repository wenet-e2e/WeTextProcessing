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

from pynini import (
    accep,
    cross,
    compose,
    difference,
    invert,
    string_file,
    string_map,
    union,
)
from pynini.examples import plurals
from pynini.lib.pynutil import delete, insert

from tn.english.rules.cardinal import Cardinal
from tn.english.rules.ordinal import Ordinal
from tn.english.rules.decimal import Decimal
from tn.english.rules.fraction import Fraction
from tn.processor import Processor
from tn.utils import get_abs_path, load_labels, get_formats

suppletive = string_file(get_abs_path("english/data/suppletive.tsv"))
# _v = union("a", "e", "i", "o", "u")
_c = union(
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
_ies = Processor.VSIGMA + _c + cross("y", "ies")
_es = Processor.VSIGMA + union("s", "sh", "ch", "x", "z") + insert("es")
_s = Processor.VSIGMA + insert("s")

graph_plural = plurals._priority_union(
    suppletive,
    plurals._priority_union(
        _ies,
        plurals._priority_union(_es, _s, Processor.VSIGMA),
        Processor.VSIGMA,
    ),
    Processor.VSIGMA,
).optimize()
SINGULAR_TO_PLURAL = graph_plural


class Measure(Processor):

    def __init__(self, deterministic: bool = False):
        """
        Args:
            deterministic: if True will provide a single transduction option,
                for False multiple transduction are generated (used for audio-based normalization)
        """
        super().__init__("measure", ordertype="en_tn")
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
            cardinal.graph_with_and
        )

        graph_unit = string_file(get_abs_path("english/data/measure/unit.tsv"))
        if not self.deterministic:
            graph_unit |= string_file(
                get_abs_path("english/data/measure/unit_alternatives.tsv")
            )

        graph_unit |= compose(
            self.TO_LOWER.plus
            + (self.ALPHA | self.TO_LOWER)
            + (self.ALPHA | self.TO_LOWER).star,
            graph_unit,
        ).optimize()

        graph_unit_plural = graph_unit @ SINGULAR_TO_PLURAL

        optional_graph_negative = (insert("negative: ") + cross("-", '"true" ')).ques

        graph_unit2 = (
            cross("/", "per") + self.DELETE_ZERO_OR_ONE_SPACE + insert(" ") + graph_unit
        )

        optional_graph_unit2 = (
            self.DELETE_ZERO_OR_ONE_SPACE + insert(" ") + graph_unit2
        ).ques

        unit_plural = (
            insert(' units: "')
            + (graph_unit_plural + optional_graph_unit2 | graph_unit2)
            + insert('"')
        )

        unit_singular = (
            insert(' units: "')
            + (graph_unit + optional_graph_unit2 | graph_unit2)
            + insert('"')
        )

        decimal = Decimal(self.deterministic)
        subgraph_decimal = (
            optional_graph_negative
            + decimal.final_graph_wo_negative
            + accep(" ").ques
            + unit_plural
        )

        # support radio FM/AM
        subgraph_decimal |= (
            decimal.final_graph_wo_negative
            + accep(" ").ques
            + insert(' units: "')
            + union("AM", "FM")
            + insert('"')
        )

        subgraph_cardinal = (
            optional_graph_negative
            + insert('integer: "')
            + ((self.VSIGMA - "1") @ cardinal_graph)
            + insert('"')
            + accep(" ").ques
            + unit_plural
        )

        subgraph_cardinal |= (
            optional_graph_negative
            + insert('integer: "')
            + cross("1", "one")
            + insert('"')
            + accep(" ").ques
            + unit_singular
        )

        unit_graph = (
            insert('integer: "-" units: "')
            + (
                (cross("/", "per") + self.DELETE_ZERO_OR_ONE_SPACE)
                | (accep("per") + delete(" "))
            )
            + insert(" ")
            + graph_unit
            + insert('"')
        )  # noqa

        decimal_dash_alpha = (
            decimal.final_graph_wo_negative
            + cross("-", "")
            + insert(' units: "')
            + self.ALPHA.plus
            + insert('"')
        )

        decimal_times = (
            decimal.final_graph_wo_negative
            + insert(' units: "')
            + (cross(union("x", "X"), "x") | cross(union("x", "X"), " times"))
            + insert('"')
        )

        alpha_dash_decimal = (
            insert('units: "')
            + self.ALPHA.plus
            + accep("-")
            + insert('"')
            + decimal.final_graph_wo_negative
        )

        fraction = Fraction(self.deterministic)
        subgraph_fraction = fraction.graph + accep(" ").ques + unit_plural

        address = self.get_address_graph(cardinal)
        address = insert('units: "address" integer: "') + address + insert('"')

        math_operations = string_file(
            get_abs_path("english/data/measure/math_operation.tsv")
        )
        delimiter = accep(" ") | insert(" ")

        math = (
            (cardinal_graph | self.ALPHA)
            + delimiter
            + math_operations
            + (delimiter | self.ALPHA)
            + cardinal_graph
            + delimiter
            + cross("=", "equals")
            + delimiter
            + (cardinal_graph | self.ALPHA)
        )

        math |= (
            (cardinal_graph | self.ALPHA)
            + delimiter
            + cross("=", "equals")
            + delimiter
            + (cardinal_graph | self.ALPHA)
            + delimiter
            + math_operations
            + delimiter
            + cardinal_graph
        )

        math = insert('units: "math" integer: "') + math + insert('"')
        final_graph = (
            subgraph_decimal
            | subgraph_cardinal
            | unit_graph
            | decimal_dash_alpha
            | decimal_times
            | alpha_dash_decimal
            | subgraph_fraction
            | address
            | math
        )

        final_graph = self.add_tokens(final_graph)
        self.tagger = final_graph.optimize()

    def get_range(self, cardinal: Processor):
        """
        Returns range forms for measure tagger, e.g. 2-3, 2x3, 2*2

        Args:
            cardinal: cardinal GraphFst
        """
        range_graph = cardinal + cross(union("-", " - "), " to ") + cardinal

        for x in [" x ", "x"]:
            range_graph |= cardinal + cross(x, " by ") + cardinal
            if not self.deterministic:
                range_graph |= cardinal + cross(x, " times ") + cardinal.ques

        for x in ["*", " * "]:
            range_graph |= cardinal + cross(x, " times ") + cardinal
        return range_graph.optimize()

    def get_address_graph(self, cardinal: Processor):
        """
        Finite state transducer for classifying serial.
            The serial is a combination of digits, letters and dashes, e.g.:
            2788 San Tomas Expy, Santa Clara, CA 95051 ->
                units: "address" integer: "two seven eight eight San Tomas Expressway Santa Clara California nine five zero five one"
        """
        ordinal = Ordinal(self.deterministic)
        ordinal_verbalizer = ordinal.graph_v
        ordinal_tagger = ordinal.graph
        ordinal_num = compose(
            insert('integer: "') + ordinal_tagger + insert('"'),
            ordinal_verbalizer,
        )

        address_num = (
            self.DIGIT ** (1, 2)
            @ cardinal.graph_hundred_component_at_least_one_none_zero_digit
        )
        address_num += insert(" ") + self.DIGIT**2 @ (
            cross("0", "zero ").ques
            + cardinal.graph_hundred_component_at_least_one_none_zero_digit
        )
        # to handle the rest of the numbers
        address_num = compose(self.DIGIT ** (3, 4), address_num)
        address_num = plurals._priority_union(address_num, cardinal.graph, self.VSIGMA)

        direction = (
            cross("E", "East")
            | cross("S", "South")
            | cross("W", "West")
            | cross("N", "North")
        ) + delete(".").ques

        direction = (accep(" ") + direction).ques
        address_words = get_formats(
            get_abs_path("english/data/address/address_word.tsv")
        )
        address_words = (
            accep(" ")
            + (ordinal_num.ques | self.UPPER + self.ALPHA.plus)
            + " "
            + (self.UPPER + self.ALPHA.star + " ").star
            + address_words
        )

        city = (self.ALPHA | accep(" ")).plus
        city = (accep(", ") + city).ques

        states = load_labels(get_abs_path("english/data/address/state.tsv"))

        additional_options = []
        for x, y in states:
            additional_options.append((x, f"{y[0]}.{y[1:]}"))
        states.extend(additional_options)
        state_graph = string_map(states)
        state = invert(state_graph)
        state = (accep(",") + accep(" ") + state).ques

        zip_code = compose(self.DIGIT**5, cardinal.single_digits)
        zip_code = accep(",").ques + accep(" ") + zip_code

        address = (
            address_num + direction + address_words + (city + state + zip_code).ques
        )

        address |= address_num + direction + address_words + cross(".", "").ques

        return address

    def build_verbalizer(self):
        """
        Finite state transducer for verbalizing measure, e.g.
            measure { negative: "true" integer: "twelve" units: "kilograms" } -> minus twelve kilograms
            measure { integer_part: "twelve" fractional_part: "five" units: "kilograms" } -> twelve point five kilograms
        """
        cardinal = Cardinal(self.deterministic)
        unit = (
            delete('units: "')
            + difference(self.NOT_QUOTE.plus, union("address", "math"))
            + delete('"')
            + self.DELETE_SPACE
        )

        if not self.deterministic:
            unit |= compose(unit, cross(union("inch", "inches"), '"'))

        decimal = Decimal(self.deterministic)
        graph_decimal = decimal.numbers

        if not self.deterministic:
            graph_decimal |= compose(
                graph_decimal,
                self.VSIGMA
                + (
                    cross(" point five", " and a half")
                    | cross("zero point five", "half")
                    | cross(" point two five", " and a quarter")
                    | cross("zero point two five", "quarter")
                ),
            ).optimize()

        graph_cardinal = cardinal.numbers

        fraction = Fraction(self.deterministic)
        graph_fraction = fraction.graph_v

        graph = (graph_cardinal | graph_decimal | graph_fraction) + accep(" ") + unit

        graph |= (
            unit + insert(" ") + (graph_cardinal | graph_decimal) + self.DELETE_SPACE
        )
        # for only unit
        graph |= delete('integer: "-"') + self.DELETE_SPACE + unit
        address = (
            delete('units: "address" ')
            + self.DELETE_SPACE
            + graph_cardinal
            + self.DELETE_SPACE
        )
        math = (
            delete('units: "math" ')
            + self.DELETE_SPACE
            + graph_cardinal
            + self.DELETE_SPACE
        )
        graph |= address | math

        delete_tokens = self.delete_tokens(graph)
        self.verbalizer = delete_tokens.optimize()
