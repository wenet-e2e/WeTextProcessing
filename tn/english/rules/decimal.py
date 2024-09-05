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
from pynini import (
    accep,
    cdrewrite,
    cross,
    compose,
    difference,
    project,
    string_file,
    union,
)
from pynini.lib.pynutil import add_weight, delete, insert

from tn.english.rules.cardinal import Cardinal
from tn.processor import Processor
from tn.utils import get_abs_path

delete_space = delete(" ")
quantities = string_file(get_abs_path("english/data/number/thousand.tsv"))
quantities_abbr = string_file(get_abs_path("english/data/number/quantity_abbr.tsv"))
quantities_abbr |= Processor.TO_UPPER @ quantities_abbr


def get_quantity(
    decimal: "pynini.FstLike",
    cardinal_up_to_hundred: "pynini.FstLike",
    include_abbr: bool,
) -> "pynini.FstLike":
    """
    Returns FST that transforms either a cardinal or decimal followed by a quantity into a numeral,
    e.g. 1 million -> integer_part: "one" quantity: "million"
    e.g. 1.5 million -> integer_part: "one" fractional_part: "five" quantity: "million"

    Args:
        decimal: decimal FST
        cardinal_up_to_hundred: cardinal FST
    """
    quantity_wo_thousand = project(quantities, "input") - union("k", "K", "thousand")
    if include_abbr:
        quantity_wo_thousand |= project(quantities_abbr, "input") - union(
            "k", "K", "thousand"
        )
    res = (
        insert('integer_part: "')
        + cardinal_up_to_hundred
        + insert('"')
        + delete(" ").ques
        + insert(' quantity: "')
        + quantity_wo_thousand @ (quantities | quantities_abbr)
        + insert('"')
    )
    if include_abbr:
        quantity = quantities | quantities_abbr
    else:
        quantity = quantities
    res |= decimal + delete(" ").ques + insert(' quantity: "') + quantity + insert('"')
    return res


class Decimal(Processor):

    def __init__(self, deterministic: bool = False):
        """
        Args:
            deterministic: if True will provide a single transduction option,
                for False multiple transduction are generated (used for audio-based normalization)
        """
        super().__init__("decimal", ordertype="en_tn")
        self.deterministic = deterministic
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        """
        Finite state transducer for classifying decimal, e.g.
            -12.5006 billion -> decimal { negative: "true" integer_part: "12" fractional_part: "five o o six" quantity: "billion" }
            1 billion -> decimal { integer_part: "one" quantity: "billion" }
        """
        cardinal = Cardinal(deterministic=self.deterministic)
        cardinal_graph = cardinal.graph_with_and
        cardinal_graph_hundred_component_at_least_one_none_zero_digit = (
            cardinal.graph_hundred_component_at_least_one_none_zero_digit
        )

        self.graph = cardinal.single_digits.optimize()

        if not self.deterministic:
            self.graph = self.graph | add_weight(cardinal_graph, 0.1)

        point = delete(".")
        optional_graph_negative = (insert("negative: ") + cross("-", '"true" ')).ques

        self.graph_fractional = insert('fractional_part: "') + self.graph + insert('"')
        self.graph_integer = insert('integer_part: "') + cardinal_graph + insert('"')
        final_graph_wo_sign = (
            (self.graph_integer + insert(" ")).ques
            + point
            + insert(" ")
            + self.graph_fractional
        )

        quantity_w_abbr = get_quantity(
            final_graph_wo_sign,
            cardinal_graph_hundred_component_at_least_one_none_zero_digit,
            include_abbr=True,
        )
        quantity_wo_abbr = get_quantity(
            final_graph_wo_sign,
            cardinal_graph_hundred_component_at_least_one_none_zero_digit,
            include_abbr=False,
        )
        self.final_graph_wo_negative_w_abbr = final_graph_wo_sign | quantity_w_abbr
        self.final_graph_wo_negative = final_graph_wo_sign | quantity_wo_abbr

        # reduce options for non_deterministic and allow either "oh" or "zero", but not combination
        if not self.deterministic:
            no_oh_zero = difference(
                self.VSIGMA,
                (self.VSIGMA + "oh" + self.VSIGMA + "zero" + self.VSIGMA)
                | (self.VSIGMA + "zero" + self.VSIGMA + "oh" + self.VSIGMA),
            ).optimize()
            no_zero_oh = difference(
                self.VSIGMA,
                self.VSIGMA + accep("zero") + self.VSIGMA + accep("oh") + self.VSIGMA,
            ).optimize()

            self.final_graph_wo_negative |= compose(
                self.final_graph_wo_negative,
                cdrewrite(
                    cross('integer_part: "zero"', 'integer_part: "oh"'),
                    self.VSIGMA,
                    self.VSIGMA,
                    self.VSIGMA,
                ),
            )
            self.final_graph_wo_negative = compose(
                self.final_graph_wo_negative, no_oh_zero
            ).optimize()
            self.final_graph_wo_negative = compose(
                self.final_graph_wo_negative, no_zero_oh
            ).optimize()

        final_graph = optional_graph_negative + self.final_graph_wo_negative

        final_graph = self.add_tokens(final_graph)
        self.tagger = final_graph.optimize()

    def build_verbalizer(self):
        """
        Finite state transducer for verbalizing decimal, e.g.
            decimal { negative: "true" integer_part: "twelve" fractional_part: "five o o six" quantity: "billion" } -> minus twelve point five o o six billion
        """
        cardinal = Cardinal(deterministic=self.deterministic)
        self.optional_sign = cross('negative: "true"', "minus ")
        if not self.deterministic:
            self.optional_sign |= add_weight(
                cross('negative: "true"', "negative "), 0.1
            )
        self.optional_sign = (self.optional_sign + self.DELETE_SPACE).ques
        self.integer = delete("integer_part:") + cardinal.integer
        self.optional_integer = (self.integer + self.DELETE_SPACE + insert(" ")).ques
        self.fractional_default = (
            delete("fractional_part:")
            + self.DELETE_SPACE
            + delete('"')
            + self.NOT_QUOTE.plus
            + delete('"')
        )

        self.fractional = insert("point ") + self.fractional_default

        self.quantity = (
            self.DELETE_SPACE
            + insert(" ")
            + delete("quantity:")
            + self.DELETE_SPACE
            + delete('"')
            + self.NOT_QUOTE.plus
            + delete('"')
        )
        self.optional_quantity = self.quantity.ques

        graph = self.optional_sign + (
            self.integer
            | (self.integer + self.quantity)
            | (self.optional_integer + self.fractional + self.optional_quantity)
        )

        self.numbers = graph
        delete_tokens = self.delete_tokens(graph)
        if not self.deterministic:
            delete_tokens |= compose(
                delete_tokens,
                self.VSIGMA
                + (
                    cross(" point five", " and a half")
                    | cross("zero point five", "half")
                    | cross(" point two five", " and a quarter")
                    | cross("zero point two five", "quarter")
                ),
            ).optimize()
        self.verbalizer = delete_tokens.optimize()
