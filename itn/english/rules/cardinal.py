# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
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

from pynini import closure, cross, difference, project, string_file, union
from pynini.lib.pynutil import delete, insert

from itn.english.rules.utils import num_to_word
from tn.processor import Processor
from tn.utils import get_abs_path


class Cardinal(Processor):
    """
    Finite state transducer for classifying cardinals
        e.g. minus twenty three -> cardinal { integer: "23" negative: "-" } }
    Numbers below thirteen are not converted.

    Args:
        input_case: accepting either "lower_cased" or "cased" input.
    """

    def __init__(self):
        super().__init__("cardinal")
        self.build_tagger()
        self.build_verbalizer()

    def delete_word(self, word: str):
        """Capitalizes word for `cased` input"""
        delete_graph = delete(word).optimize()
        if len(word) > 0:
            delete_graph |= delete(word[0].upper() + word[1:])
        return delete_graph.optimize()

    def build_tagger(self):
        graph_zero = string_file(get_abs_path("../itn/english/data/numbers/zero.tsv"))
        graph_digit = string_file(get_abs_path("../itn/english/data/numbers/digit.tsv"))
        graph_ties = string_file(get_abs_path("../itn/english/data/numbers/ties.tsv"))
        graph_teen = string_file(get_abs_path("../itn/english/data/numbers/teen.tsv"))
        self.graph_two_digit = graph_teen | ((graph_ties) + self.DELETE_SPACE + (graph_digit | insert("0")))
        graph_hundred = cross("hundred", "")

        graph_hundred_component = union(graph_digit + self.DELETE_SPACE + graph_hundred, insert("0"))
        graph_hundred_component += self.DELETE_SPACE
        graph_hundred_component += union(
            graph_teen | insert("00"), (graph_ties | insert("0")) + self.DELETE_SPACE + (graph_digit | insert("0"))
        )

        graph_hundred_component_at_least_one_none_zero_digit = graph_hundred_component @ (
            closure(self.DIGIT) + (self.DIGIT - "0") + closure(self.DIGIT)
        )
        self.graph_hundred_component_at_least_one_none_zero_digit = graph_hundred_component_at_least_one_none_zero_digit

        # Transducer for eleven hundred -> 1100 or twenty one hundred eleven -> 2111
        graph_hundred_as_thousand = union(graph_teen, graph_ties + self.DELETE_SPACE + graph_digit)
        graph_hundred_as_thousand += self.DELETE_SPACE + graph_hundred
        graph_hundred_as_thousand += self.DELETE_SPACE + union(
            graph_teen | insert("00"),
            (graph_ties | insert("0")) + self.DELETE_SPACE + (graph_digit | insert("0")),
        )

        graph_hundreds = graph_hundred_component | graph_hundred_as_thousand

        graph_ties_component = union(
            graph_teen | insert("00"),
            (graph_ties | insert("0")) + self.DELETE_SPACE + (graph_digit | insert("0")),
        )

        graph_ties_component_at_least_one_none_zero_digit = graph_ties_component @ (
            closure(self.DIGIT) + (self.DIGIT - "0") + closure(self.DIGIT)
        )
        self.graph_ties_component_at_least_one_none_zero_digit = graph_ties_component_at_least_one_none_zero_digit

        # %%% International numeric format
        graph_thousands = union(
            graph_hundred_component_at_least_one_none_zero_digit + self.DELETE_SPACE + self.delete_word("thousand"),
            insert("000", weight=0.1),
        )

        graph_million = union(
            graph_hundred_component_at_least_one_none_zero_digit + self.DELETE_SPACE + self.delete_word("million"),
            insert("000", weight=0.1),
        )
        graph_billion = union(
            graph_hundred_component_at_least_one_none_zero_digit + self.DELETE_SPACE + self.delete_word("billion"),
            insert("000", weight=0.1),
        )
        graph_trillion = union(
            graph_hundred_component_at_least_one_none_zero_digit + self.DELETE_SPACE + self.delete_word("trillion"),
            insert("000", weight=0.1),
        )
        graph_quadrillion = union(
            graph_hundred_component_at_least_one_none_zero_digit + self.DELETE_SPACE + self.delete_word("quadrillion"),
            insert("000", weight=0.1),
        )
        graph_quintillion = union(
            graph_hundred_component_at_least_one_none_zero_digit + self.DELETE_SPACE + self.delete_word("quintillion"),
            insert("000", weight=0.1),
        )
        graph_sextillion = union(
            graph_hundred_component_at_least_one_none_zero_digit + self.DELETE_SPACE + self.delete_word("sextillion"),
            insert("000", weight=0.1),
        )
        # %%%

        graph_int = (
            graph_sextillion
            + self.DELETE_SPACE
            + graph_quintillion
            + self.DELETE_SPACE
            + graph_quadrillion
            + self.DELETE_SPACE
            + graph_trillion
            + self.DELETE_SPACE
            + graph_billion
            + self.DELETE_SPACE
            + graph_million
            + self.DELETE_SPACE
            + graph_thousands
        )

        # %% Indian numeric format simple https://en.wikipedia.org/wiki/Indian_numbering_system
        # This only covers "standard format".
        # Conventional format like thousand crores/lakh crores is yet to be implemented
        graph_in_thousands = union(
            graph_ties_component_at_least_one_none_zero_digit + self.DELETE_SPACE + delete("thousand"),
            insert("00", weight=0.1),
        )
        graph_in_lakhs = union(
            graph_ties_component_at_least_one_none_zero_digit + self.DELETE_SPACE + (delete("lakh") | delete("lakhs")),
            insert("00", weight=0.1),
        )

        graph_in_crores = union(
            graph_ties_component_at_least_one_none_zero_digit
            + self.DELETE_SPACE
            + (delete("crore") | delete("crores")),
            insert("00", weight=0.1),
        )

        graph_in_arabs = union(
            graph_ties_component_at_least_one_none_zero_digit + self.DELETE_SPACE + (delete("arab") | delete("arabs")),
            insert("00", weight=0.1),
        )

        graph_in_kharabs = union(
            graph_ties_component_at_least_one_none_zero_digit
            + self.DELETE_SPACE
            + (delete("kharab") | delete("kharabs")),
            insert("00", weight=0.1),
        )

        graph_in_nils = union(
            graph_ties_component_at_least_one_none_zero_digit + self.DELETE_SPACE + (delete("nil") | delete("nils")),
            insert("00", weight=0.1),
        )

        graph_in_padmas = union(
            graph_ties_component_at_least_one_none_zero_digit
            + self.DELETE_SPACE
            + (delete("padma") | delete("padmas")),
            insert("00", weight=0.1),
        )

        graph_in_shankhs = union(
            graph_ties_component_at_least_one_none_zero_digit
            + self.DELETE_SPACE
            + (delete("shankh") | delete("shankhs")),
            insert("00", weight=0.1),
        )

        graph_ind = (
            graph_in_shankhs
            + self.DELETE_SPACE
            + graph_in_padmas
            + self.DELETE_SPACE
            + graph_in_nils
            + self.DELETE_SPACE
            + graph_in_kharabs
            + self.DELETE_SPACE
            + graph_in_arabs
            + self.DELETE_SPACE
            + graph_in_crores
            + self.DELETE_SPACE
            + graph_in_lakhs
            + self.DELETE_SPACE
            + graph_in_thousands
        )

        graph = union((graph_int | graph_ind) + self.DELETE_SPACE + graph_hundreds, graph_zero)
        graph = graph @ union(delete(closure("0")) + difference(self.DIGIT, "0") + closure(self.DIGIT), "0")

        labels_exception = [num_to_word(x) for x in range(0, 13)]
        labels_exception += [x.capitalize() for x in labels_exception]
        graph_exception = union(*labels_exception).optimize()

        graph = (self.build_rule(delete("and"), self.SPACE, self.SPACE) @ (self.ALPHA + self.SIGMA) @ graph).optimize()
        graph |= ((self.TO_LOWER + self.SIGMA) @ graph).optimize()

        self.graph_no_exception = graph
        self.graph = (project(graph, "input") - graph_exception.arcsort()) @ graph
        tagger = (
            insert('value: "') + cross(union("minus", "Minus") + self.DELETE_SPACE, "-").ques + self.graph + insert('"')
        )
        self.tagger = self.add_tokens(tagger).optimize()
