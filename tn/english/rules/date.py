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
from tn.english.rules.ordinal import Ordinal
from tn.processor import Processor
from tn.utils import augment_labels_with_punct_at_end, get_abs_path, load_labels

graph_teen = pynini.invert(pynini.string_file(get_abs_path("english/data/number/teen.tsv"))).optimize()
graph_digit = pynini.invert(pynini.string_file(get_abs_path("english/data/number/digit.tsv"))).optimize()
ties_graph = pynini.invert(pynini.string_file(get_abs_path("english/data/number/ty.tsv"))).optimize()
year_suffix = load_labels(get_abs_path("english/data/date/year_suffix.tsv"))
year_suffix.extend(augment_labels_with_punct_at_end(year_suffix))
year_suffix = pynini.string_map(year_suffix).optimize()


def get_ties_graph(deterministic: bool = False):
    """
    Returns two digit transducer, e.g.
    03 -> o three
    12 -> thirteen
    20 -> twenty
    """
    graph = graph_teen | ties_graph + pynutil.delete("0") | ties_graph + Processor("tmp").INSERT_SPACE + graph_digit

    if deterministic:
        graph = graph | pynini.cross("0", "oh") + Processor("tmp").INSERT_SPACE + graph_digit
    else:
        graph = (graph | (pynini.cross("0", "oh") | pynini.cross("0", "zero")) + Processor("tmp").INSERT_SPACE + graph_digit)

    return graph.optimize()


def get_four_digit_year_graph(deterministic: bool = False):
    """
    Returns a four digit transducer which is combination of ties/teen or digits
    (using hundred instead of thousand format), e.g.
    1219 -> twelve nineteen
    3900 -> thirty nine hundred
    """
    graph_ties = get_ties_graph(deterministic)

    graph_with_s = ((graph_ties + Processor("tmp").INSERT_SPACE + graph_ties)
                    | (graph_teen + Processor("tmp").INSERT_SPACE +
                       (ties_graph | pynini.cross("1", "ten")))) + pynutil.delete("0s")

    graph_with_s |= ((graph_teen | graph_ties) + Processor("tmp").INSERT_SPACE + pynini.cross("00", "hundred") +
                     pynutil.delete("s"))
    graph_with_s = (graph_with_s @ pynini.cdrewrite(
        pynini.cross("y", "ies") | pynutil.insert("s"),
        "",
        "[EOS]",
        Processor("tmp").VCHAR,
    ).star)

    graph = graph_ties + Processor("tmp").INSERT_SPACE + graph_ties
    graph |= (graph_teen | graph_ties) + Processor("tmp").INSERT_SPACE + pynini.cross("00", "hundred")

    thousand_graph = (graph_digit + Processor("tmp").INSERT_SPACE + pynini.cross("00", "thousand") +
                      (pynutil.delete("0") | Processor("tmp").INSERT_SPACE + graph_digit))
    thousand_graph |= (graph_digit + Processor("tmp").INSERT_SPACE + pynini.cross("000", "thousand") +
                       pynutil.delete(" ").ques + pynini.accep("s"))

    graph |= graph_with_s
    if deterministic:
        graph = plurals._priority_union(thousand_graph, graph, Processor("tmp").VCHAR.star)
    else:
        graph |= thousand_graph

    return graph.optimize()


def _get_two_digit_year_with_s_graph():
    # to handle '70s -> seventies
    graph = (pynutil.delete("'").ques + pynini.compose(
        ties_graph + pynutil.delete("0s"),
        pynini.cdrewrite(pynini.cross("y", "ies"), "", "[EOS]",
                         Processor("tmp").VCHAR.star),
    )).optimize()
    return graph


def _get_year_graph(cardinal_graph, deterministic: bool = False):
    """
    Transducer for year, only from 1000 - 2999 e.g.
    1290 -> twelve nineteen
    2000 - 2009 will be verbalized as two thousand.

    Transducer for 3 digit year, e.g. 123-> one twenty three

    Transducer for year with suffix
    123 A.D., 4200 B.C
    """
    graph = get_four_digit_year_graph(deterministic)
    graph = (pynini.union("1", "2") + (Processor("tmp").DIGIT**3) + (pynini.cross(" s", "s") | "s").ques) @ graph

    graph |= _get_two_digit_year_with_s_graph()

    three_digit_year = ((Processor("tmp").DIGIT @ cardinal_graph) + Processor("tmp").INSERT_SPACE +
                        (Processor("tmp").DIGIT**2) @ cardinal_graph)
    year_with_suffix = ((get_four_digit_year_graph(deterministic=True) | three_digit_year) + Processor("tmp").DELETE_SPACE +
                        Processor("tmp").INSERT_SPACE + year_suffix)
    graph |= year_with_suffix
    return graph.optimize()


def _get_two_digit_year(cardinal_graph, single_digits_graph):
    two_digit_year = Processor("tmp").DIGIT**(2) @ plurals._priority_union(cardinal_graph, single_digits_graph,
                                                                           Processor("tmp").VCHAR.star)
    return two_digit_year


def _get_financial_period_graph():
    # 1H23 -> first half of twenty three
    # 3Q22 -> third quarter of twenty two

    h_ordinals = pynini.cross("1", "first") | pynini.cross("2", "second")
    q_ordinals = h_ordinals | pynini.cross("3", "third") | pynini.cross("4", "fourth")

    h_graph = h_ordinals + pynini.cross("H", " half")
    q_graph = q_ordinals + pynini.cross("Q", " quarter")
    period_graph = h_graph | q_graph

    return period_graph


class Date(Processor):

    def __init__(self, deterministic: bool = False, cardinal=None, ordinal=None):
        super().__init__("date", ordertype="en_tn")
        self.deterministic = deterministic
        self.cardinal = cardinal or Cardinal(deterministic)
        self.ordinal = ordinal or Ordinal(deterministic, cardinal=self.cardinal)
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        """
        Finite state transducer for classifying date, e.g.
            jan. 5, 2012 -> date { month: "jan." day: "5" year: ", 2012" }
            jan. 5 -> date { month: "jan." day: "5" }
            5 january 2012 -> date { day: "5" month: "january" year: "2012" }
            2012-01-05 -> date { year: "2012" month: "01" day: "05" }
            2012.01.05 -> date { year: "2012" month: "01" day: "05" }
            2012/01/05 -> date { year: "2012" month: "01" day: "05" }
            2012 -> date { year: "2012" }
        """
        cardinal = self.cardinal
        # january, January, JANUARY
        month_graph = pynini.string_file(get_abs_path("english/data/date/month_name.tsv"))
        # jan, Jan, JAN
        month_abbr_graph = pynini.string_file(get_abs_path("english/data/date/month_abbr.tsv"))
        month_graph |= month_abbr_graph
        month_graph += pynutil.delete(self.PUNCT).ques

        month_numbers_labels = pynini.string_file(get_abs_path("english/data/date/month_number.tsv"))
        cardinal_graph = cardinal.graph_hundred_component_at_least_one_none_zero_digit

        year_graph = _get_year_graph(cardinal_graph=cardinal_graph, deterministic=self.deterministic)

        # three_digit_year = (self.DIGIT @ cardinal_graph) + self.INSERT_SPACE + (self.DIGIT ** 2) @ cardinal_graph
        # year_graph |= three_digit_year

        self.month_graph = month_graph
        self.month_numbers_graph = month_numbers_labels
        month_graph = self.tag_field("month", self.month_graph)
        month_numbers_graph = self.tag_field("month", self.month_numbers_graph)

        endings = ["rd", "th", "st", "nd"]
        endings += [x.upper() for x in endings]
        endings = pynini.union(*endings)

        day_number = (pynini.union("1", "2") + self.DIGIT) | self.DIGIT | (pynini.accep("3") + pynini.union("0", "1"))
        self.day_graph = (pynutil.delete("the ").ques +
                          (pynutil.delete("0").ques + day_number + pynutil.delete(endings).ques) @ cardinal_graph)
        day_graph = self.tag_field("day", self.day_graph)

        two_digit_year = _get_two_digit_year(
            cardinal_graph=cardinal_graph,
            single_digits_graph=cardinal.single_digits_graph,
        )
        self.two_digit_year_graph = two_digit_year + pynutil.delete(",").ques
        two_digit_year = self.tag_field("year", self.two_digit_year_graph)

        self.year_after_graph = (pynutil.delete(" ") + year_graph + pynutil.delete(",").ques
                                 | pynini.cross(",", ", ") + pynutil.delete(" ").ques + year_graph + pynutil.delete(",").ques)
        graph_year = self.INSERT_SPACE + self.tag_field("year", self.year_after_graph)
        optional_graph_year = graph_year.ques

        self.year_graph = year_graph
        year_graph = self.tag_field("year", self.year_graph)

        graph_mdy = month_graph + ((self.DELETE_EXTRA_SPACE + day_graph)
                                   | (pynini.accep(" ") + day_graph)
                                   | graph_year
                                   | (self.DELETE_EXTRA_SPACE + day_graph + graph_year))
        graph_mdy |= (month_graph + pynini.cross("-", " ") + day_graph +
                      ((pynini.cross("-", " ") + self.VCHAR.star) @ graph_year).ques)
        for x in ["-", "/", "."]:
            delete_sep = pynutil.delete(x)
            graph_mdy |= (month_numbers_graph + delete_sep + self.INSERT_SPACE + day_graph + delete_sep + self.INSERT_SPACE +
                          (pynutil.add_weight(year_graph, -1.0)))

        graph_dmy = day_graph + self.DELETE_EXTRA_SPACE + self.INSERT_SPACE + month_graph + optional_graph_year
        day_ex_month = (self.DIGIT**2 - pynini.project(month_numbers_graph, "input")) @ day_graph
        for x in ["-", "/", "."]:
            delete_sep = pynutil.delete(x)
            graph_dmy |= (day_ex_month + delete_sep + self.INSERT_SPACE + month_numbers_graph + delete_sep +
                          self.INSERT_SPACE + (pynutil.add_weight(year_graph, -1.0)))

        graph_ymd = (year_graph + self.DELETE_EXTRA_SPACE + self.INSERT_SPACE + month_graph + self.DELETE_EXTRA_SPACE +
                     self.INSERT_SPACE + day_graph)
        for x in ["-", "/", "."]:
            delete_sep = pynutil.delete(x)
            graph_ymd |= ((pynutil.add_weight(year_graph, -1.0)) + delete_sep + self.INSERT_SPACE + month_numbers_graph +
                          delete_sep + self.INSERT_SPACE + day_graph)

        final_graph = pynutil.add_weight(graph_mdy | graph_dmy | graph_ymd, -0.1) | year_graph

        self.period_graph = _get_financial_period_graph()
        period_fy = self.tag_field("text", self.period_graph)
        graph_fy = period_fy + self.INSERT_SPACE + two_digit_year

        final_graph |= graph_fy

        self.structured_graph = final_graph.optimize()
        self.tagger = self.add_tokens(self.structured_graph)

    def build_verbalizer(self):
        """
        Finite state transducer for verbalizing date, e.g.
            date { month: "february" day: "5" year: "2012" } -> february fifth twenty twelve
            date { day: "5" month: "february" year: "2012" } -> the fifth of february twenty twelve
        """
        ordinal = self.ordinal
        day_cardinal = (pynutil.delete("day:") + self.DELETE_SPACE + pynutil.delete('"') + self.day_graph +
                        pynutil.delete('"'))
        day = day_cardinal @ ordinal.suffix
        period = (pynutil.delete("text:") + self.DELETE_SPACE + pynutil.delete('"') + self.period_graph + pynutil.delete('"'))
        month = (pynutil.delete("month:") + self.DELETE_SPACE + pynutil.delete('"') +
                 (self.month_graph | self.month_numbers_graph) + pynutil.delete('"'))

        year = (pynutil.delete("year:") + self.DELETE_SPACE + pynutil.delete('"') +
                (self.year_graph | self.year_after_graph | self.two_digit_year_graph) + self.DELETE_SPACE +
                pynutil.delete('"'))

        # financial period
        graph_fy = pynutil.insert("the ") + period + pynutil.insert(" of") + (self.DELETE_EXTRA_SPACE + year).ques

        # day month year, month year
        graph_dmy = ((pynutil.insert("the ") + day + self.DELETE_EXTRA_SPACE + pynutil.insert("of ")).ques + month +
                     (self.DELETE_EXTRA_SPACE + year).ques)

        final_graph = (graph_dmy | year | graph_fy) + self.DELETE_SPACE
        final_graph = final_graph @ self.build_rule(pynini.cross(" ,", ","))
        self.verbalizer_graph = final_graph.optimize()
        self.graph = (self.structured_graph @ self.verbalizer_graph).optimize()
        self.verbalizer = self.delete_tokens(self.verbalizer_graph)
