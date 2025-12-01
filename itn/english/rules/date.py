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

from pynini import accep, closure, cross, string_file, union
from pynini.lib.pynutil import add_weight, delete, insert

from itn.english.rules.ordinal import Ordinal
from tn.processor import Processor
from tn.utils import get_abs_path


class Date(Processor):
    """
    Finite state transducer for classifying date,
        e.g. january fifth twenty twelve -> date { month: "january" day: "5" year: "2012" preserve_order: true }
        e.g. the fifth of january twenty twelve -> date { day: "5" month: "january" year: "2012" preserve_order: true }
        e.g. twenty twenty -> date { year: "2012" preserve_order: true }

    Args:
        ordinal: OrdinalFst
        input_case: accepting either "lower_cased" or "cased" input.
    """

    def __init__(self):
        super().__init__("date")
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        graph_teen = string_file(get_abs_path("../itn/english/data/numbers/teen.tsv")).optimize()
        graph_digit = string_file(get_abs_path("../itn/english/data/numbers/digit.tsv")).optimize()
        ties_graph = string_file(get_abs_path("../itn/english/data/numbers/ties.tsv")).optimize()
        month_graph = string_file(get_abs_path("../itn/english/data/months.tsv"))
        financial_period_graph = string_file(get_abs_path("../itn/english/data/date_period.tsv")).invert()
        graph_ad_bc = self.DELETE_SPACE + string_file(get_abs_path("../itn/english/data/year_suffix.tsv")).invert()

        def _get_ties_graph():
            """
            Transducer for 20-99 e.g
            twenty three -> 23
            """
            graph = ties_graph + (self.DELETE_SPACE + self.DIGIT | insert("0"))
            graph |= ((self.TO_LOWER + self.SIGMA) @ graph).optimize()
            return graph

        def _get_range_graph():
            """
            Transducer for decades (1**0s, 2**0s), centuries (2*00s, 1*00s), millennia (2000s)
            """
            graph_ties = _get_ties_graph()
            graph = (graph_ties | graph_teen) + self.DELETE_SPACE + cross("hundreds", "00s")
            graph |= cross("two", "2") + self.DELETE_SPACE + cross("thousands", "000s")
            graph |= (
                (graph_ties | graph_teen)
                + self.DELETE_SPACE
                + (closure(self.ALPHA, 1) + (cross("ies", "y") | delete("s"))) @ (graph_ties | cross("ten", "10"))
                + insert("s")
            )
            graph @= union("1", "2") + self.DIGIT + self.DIGIT + self.DIGIT + "s"
            graph |= ((self.TO_LOWER + self.SIGMA) @ graph).optimize()
            return graph

        def _get_year_graph():
            """
            Transducer for year, e.g. twenty twenty -> 2020
            """

            def _get_digits_graph():
                zero = cross((accep("oh") | accep("o")), "0")
                graph = zero + self.DELETE_SPACE + graph_digit
                graph |= ((self.TO_LOWER + self.SIGMA) @ graph).optimize()
                return graph

            def _get_thousands_graph():
                graph_ties = _get_ties_graph()
                graph_hundred_component = (graph_digit + self.DELETE_SPACE + delete("hundred")) | insert("0")
                optional_end = closure(delete("and "), 0, 1)
                graph = (
                    graph_digit
                    + self.DELETE_SPACE
                    + delete("thousand")
                    + self.DELETE_SPACE
                    + graph_hundred_component
                    + self.DELETE_SPACE
                    + (graph_teen | graph_ties | (optional_end + insert("0") + graph_digit))
                )
                graph |= ((self.TO_LOWER + self.SIGMA) @ graph).optimize()
                return graph

            graph_ties = _get_ties_graph()
            graph_digits = _get_digits_graph()
            graph_thousands = _get_thousands_graph()

            year_graph = (
                # 20 19, 40 12, 2012 - assuming no limit on the year
                (graph_teen + self.DELETE_SPACE + (graph_ties | graph_digits | graph_teen))
                | (graph_ties + self.DELETE_SPACE + (graph_ties | graph_digits | graph_teen))
                | graph_thousands
                | ((graph_digit + self.DELETE_SPACE + (graph_ties | graph_digits | graph_teen)) + graph_ad_bc)
                | (
                    (graph_digit | graph_teen | graph_digits | graph_ties | graph_thousands)
                    + self.DELETE_SPACE
                    + graph_ad_bc
                )
                | (
                    (graph_ties + self.DELETE_SPACE + (graph_ties | graph_digits | graph_teen))
                    + self.DELETE_SPACE
                    + graph_ad_bc
                )
                | (
                    (
                        (graph_teen | graph_digit)
                        + self.DELETE_SPACE
                        + delete("hundred")
                        + insert("0")
                        + (graph_digit | insert("0"))
                    )
                    + self.DELETE_SPACE
                    + graph_ad_bc
                )
            )
            year_graph.optimize()
            year_graph |= ((self.TO_LOWER + self.SIGMA) @ year_graph).optimize()
            return year_graph

        year_graph = _get_year_graph()
        YEAR_WEIGHT = 0.001
        year_graph = add_weight(year_graph, YEAR_WEIGHT)
        month_graph = insert('month: "') + month_graph + insert('"')

        day_graph = insert('day: "') + add_weight(Ordinal().graph, -0.7) + insert('"')
        graph_year = self.DELETE_EXTRA_SPACE + insert('year: "') + add_weight(year_graph, -YEAR_WEIGHT) + insert('"')
        optional_graph_year = closure(graph_year, 0, 1)
        graph_mdy = month_graph + (
            (self.DELETE_EXTRA_SPACE + day_graph) | graph_year | (self.DELETE_EXTRA_SPACE + day_graph + graph_year)
        )
        the_graph = delete("the")
        the_graph |= delete("The").optimize()

        graph_dmy = (
            the_graph
            + self.DELETE_SPACE
            + day_graph
            + self.DELETE_SPACE
            + delete("of")
            + self.DELETE_EXTRA_SPACE
            + month_graph
            + optional_graph_year
        )

        period_fy = insert('text: "') + financial_period_graph + (cross(" ", "") | cross(" of ", "")) + insert('"')
        graph_year = insert('year: "') + (year_graph | _get_range_graph()) + insert('"')
        graph_fy = period_fy + insert(" ") + graph_year
        tagger = graph_mdy | graph_dmy | graph_year | graph_fy
        tagger += insert(" preserve_order: \"true\"")
        self.tagger = self.add_tokens(tagger).optimize()

    def build_verbalizer(self):
        month = delete('month: "') + closure(self.NOT_QUOTE, 1) + delete('"')
        day = delete('day: "') + closure(self.NOT_QUOTE, 1) + delete('"')
        year = delete('year: "') + closure(self.NOT_QUOTE, 1) + delete('"')
        period = delete('text: "') + closure(self.NOT_QUOTE, 1) + delete('"')
        graph_fy = period + closure(self.DELETE_EXTRA_SPACE + year, 0, 1)
        # month (day) year
        graph_mdy = month + closure(self.DELETE_EXTRA_SPACE + day, 0, 1) + closure(self.DELETE_EXTRA_SPACE + year, 0, 1)
        # (day) month year
        graph_dmy = closure(day + self.DELETE_EXTRA_SPACE, 0, 1) + month + closure(self.DELETE_EXTRA_SPACE + year, 0, 1)

        verbalizer = graph_mdy | year | graph_dmy | graph_fy
        verbalizer = (graph_mdy | year | graph_dmy | graph_fy) + delete(" preserve_order: \"true\"")
        self.verbalizer = self.delete_tokens(verbalizer).optimize()
