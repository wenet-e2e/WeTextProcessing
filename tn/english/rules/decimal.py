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
from pynini.lib import pynutil

from tn.english.rules.cardinal import Cardinal
from tn.processor import Processor
from tn.utils import get_abs_path

delete_space = pynutil.delete(" ")
quantities = pynini.string_file(get_abs_path("english/data/number/thousand.tsv"))
quantities_abbr = pynini.string_file(get_abs_path("english/data/number/quantity_abbr.tsv"))
quantities_abbr |= Processor("tmp").TO_UPPER @ quantities_abbr


def get_quantity(
    decimal: "pynini.FstLike",
    cardinal_up_to_hundred: "pynini.FstLike",
    include_abbr: bool,
) -> "pynini.FstLike":
    """
    Returns FST that transforms a cardinal or decimal followed by a quantity into raw fields,
    e.g. 1 million -> integer_part: "1" quantity: "million"
    e.g. 1.5 million -> integer_part: "1" fractional_part: "5" quantity: "million"

    Args:
        decimal: decimal FST
        cardinal_up_to_hundred: cardinal FST
    """
    quantity_wo_thousand = pynini.project(quantities, "input") - pynini.union("k", "K", "thousand")
    if include_abbr:
        quantity_wo_thousand |= pynini.project(quantities_abbr, "input") - pynini.union("k", "K", "thousand")
    processor = Processor("tmp")
    res = (processor.tag_field("integer_part", cardinal_up_to_hundred) + pynutil.delete(" ").ques + pynutil.insert(" ") +
           processor.tag_field("quantity", quantity_wo_thousand @ (quantities | quantities_abbr)))
    if include_abbr:
        quantity = quantities | quantities_abbr
    else:
        quantity = quantities
    res |= (decimal + pynutil.delete(" ").ques + pynutil.insert(" ") + processor.tag_field("quantity", quantity))
    return res


class Decimal(Processor):

    def __init__(self, deterministic: bool = False, cardinal=None):
        super().__init__("decimal", ordertype="en_tn")
        self.deterministic = deterministic
        self.cardinal = cardinal or Cardinal(deterministic)
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        """
        Finite state transducer for classifying decimal, e.g.
            -12.5006 billion -> decimal { negative: "-" integer_part: "12" fractional_part: "5006" quantity: "billion" }
            1 billion -> decimal { integer_part: "1" quantity: "billion" }
        """
        cardinal = self.cardinal
        cardinal_graph = cardinal.graph_with_and
        cardinal_graph_hundred_component_at_least_one_none_zero_digit = (
            cardinal.graph_hundred_component_at_least_one_none_zero_digit)

        self.graph = cardinal.single_digits_graph.optimize()

        if not self.deterministic:
            self.graph = self.graph | pynutil.add_weight(cardinal_graph, 0.1)

        point = pynutil.delete(".")
        optional_graph_negative = (self.tag_field("negative", pynini.cross("-", "true")) + self.INSERT_SPACE).ques

        self.graph_fractional = self.tag_field("fractional_part", self.graph)
        self.graph_integer = self.tag_field("integer_part", cardinal_graph)
        final_graph_wo_sign = ((self.graph_integer + pynutil.insert(" ")).ques + point + pynutil.insert(" ") +
                               self.graph_fractional)

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
        self.quantity_graph = quantities | quantities_abbr

        final_graph = optional_graph_negative + self.final_graph_wo_negative

        final_graph = self.add_tokens(final_graph)
        self.tagger = final_graph.optimize()

    def build_verbalizer(self):
        """
        Finite state transducer for verbalizing decimal, e.g.
            decimal { negative: "-" integer_part: "12" fractional_part: "5006" quantity: "billion" } -> minus twelve point five zero zero six billion
        """
        cardinal = self.cardinal
        self.optional_sign = pynini.cross('negative: "-"', "minus ")
        if not self.deterministic:
            self.optional_sign |= pynutil.add_weight(pynini.cross('negative: "-"', "negative "), 0.1)
        self.optional_sign = (self.optional_sign + self.DELETE_SPACE).ques
        self.integer = pynutil.delete("integer_part:") + cardinal.integer
        self.optional_integer = (self.integer + self.DELETE_SPACE + self.INSERT_SPACE).ques
        self.fractional_default = (pynutil.delete("fractional_part:") + self.DELETE_SPACE + pynutil.delete('"') + self.graph +
                                   pynutil.delete('"'))

        self.fractional = pynutil.insert("point ") + self.fractional_default

        self.quantity = (self.DELETE_SPACE + self.INSERT_SPACE + pynutil.delete("quantity:") + self.DELETE_SPACE +
                         pynutil.delete('"') + self.quantity_graph + pynutil.delete('"'))
        self.optional_quantity = self.quantity.ques

        graph = self.optional_sign + (self.integer
                                      | (self.integer + self.quantity)
                                      | (self.optional_integer + self.fractional + self.optional_quantity))

        self.numbers = graph
        delete_tokens = self.delete_tokens(graph)
        if not self.deterministic:
            colloquial_fraction = pynini.compose(
                delete_tokens,
                self.VCHAR.star + (pynini.cross(" point five", " and a half")
                                   | pynini.cross("zero point five", "half")
                                   | pynini.cross(" point two five", " and a quarter")
                                   | pynini.cross("zero point two five", "quarter")),
            ).optimize()
            delete_tokens |= pynutil.add_weight(colloquial_fraction, 0.0001)
        self.verbalizer = delete_tokens.optimize()
