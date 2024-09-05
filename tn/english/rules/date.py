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
    cdrewrite,
    compose,
    cross,
    invert,
    project,
    string_file,
    string_map,
    union,
)
from pynini.examples import plurals
from pynini.lib.pynutil import add_weight, delete, insert

from tn.english.rules.cardinal import Cardinal
from tn.english.rules.ordinal import Ordinal
from tn.processor import Processor
from tn.utils import get_abs_path, load_labels, augment_labels_with_punct_at_end

graph_teen = invert(
    string_file(get_abs_path("english/data/number/teen.tsv"))
).optimize()
graph_digit = invert(
    string_file(get_abs_path("english/data/number/digit.tsv"))
).optimize()
ties_graph = invert(string_file(get_abs_path("english/data/number/ty.tsv"))).optimize()
year_suffix = load_labels(get_abs_path("english/data/date/year_suffix.tsv"))
year_suffix.extend(augment_labels_with_punct_at_end(year_suffix))
year_suffix = string_map(year_suffix).optimize()


def get_ties_graph(deterministic: bool = False):
    """
    Returns two digit transducer, e.g.
    03 -> o three
    12 -> thirteen
    20 -> twenty
    """
    graph = (
        graph_teen | ties_graph + delete("0") | ties_graph + insert(" ") + graph_digit
    )

    if deterministic:
        graph = graph | cross("0", "oh") + insert(" ") + graph_digit
    else:
        graph = (
            graph | (cross("0", "oh") | cross("0", "zero")) + insert(" ") + graph_digit
        )

    return graph.optimize()


def get_four_digit_year_graph(deterministic: bool = False):
    """
    Returns a four digit transducer which is combination of ties/teen or digits
    (using hundred instead of thousand format), e.g.
    1219 -> twelve nineteen
    3900 -> thirty nine hundred
    """
    graph_ties = get_ties_graph(deterministic)

    graph_with_s = (
        (graph_ties + insert(" ") + graph_ties)
        | (graph_teen + insert(" ") + (ties_graph | cross("1", "ten")))
    ) + delete("0s")

    graph_with_s |= (graph_teen | graph_ties) + cross("00s", " hundred")
    graph_with_s = (
        graph_with_s
        @ cdrewrite(
            cross("y", "ies") | insert("s"),
            "",
            "[EOS]",
            Processor.VCHAR,
        ).star
    )

    graph = graph_ties + insert(" ") + graph_ties
    graph |= (graph_teen | graph_ties) + cross("00", " hundred")

    thousand_graph = (
        graph_digit
        + cross("00", " thousand")
        + (delete("0") | insert(" ") + graph_digit)
    )
    thousand_graph |= (
        graph_digit + cross("000", " thousand") + delete(" ").ques + accep("s")
    )

    graph |= graph_with_s
    if deterministic:
        graph = plurals._priority_union(thousand_graph, graph, Processor.VSIGMA)
    else:
        graph |= thousand_graph

    return graph.optimize()


def _get_two_digit_year_with_s_graph():
    # to handle '70s -> seventies
    graph = (
        delete("'").ques
        + compose(
            ties_graph + delete("0s"),
            cdrewrite(cross("y", "ies"), "", "[EOS]", Processor.VSIGMA),
        )
    ).optimize()
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
    graph = (
        union("1", "2") + Processor.DIGIT**3 + (cross(" s", "s") | accep("s")).ques
    ) @ graph

    graph |= _get_two_digit_year_with_s_graph()

    three_digit_year = (
        Processor.DIGIT @ cardinal_graph
        + insert(" ")
        + Processor.DIGIT**2 @ cardinal_graph
    )
    year_with_suffix = (
        (get_four_digit_year_graph(deterministic) | three_digit_year)
        + Processor.DELETE_EXTRA_SPACE
        + year_suffix
    )
    graph |= year_with_suffix
    return graph.optimize()


def _get_two_digit_year(cardinal_graph, single_digits):
    two_digit_year = Processor.DIGIT**2 @ plurals._priority_union(
        cardinal_graph, single_digits, Processor.VSIGMA
    )
    return two_digit_year


def _get_financial_period_graph():
    # 1H23 -> first half of twenty three
    # 3Q22 -> third quarter of twenty two

    h_ordinals = cross("1", "first") | cross("2", "second")
    q_ordinals = h_ordinals | cross("3", "third") | cross("4", "fourth")

    h_graph = h_ordinals + cross("H", " half")
    q_graph = q_ordinals + cross("Q", " quarter")
    period_graph = h_graph | q_graph

    return period_graph


class Date(Processor):

    def __init__(self, deterministic: bool = False):
        """
        Args:
            deterministic: if True will provide a single transduction option,
                for False multiple transduction are generated (used for audio-based normalization)
        """
        super().__init__("date", ordertype="en_tn")
        self.deterministic = deterministic
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        """
        Finite state transducer for classifying date, e.g.
            jan. 5, 2012 -> date { month: "january" day: "five" year: ", twenty twelve" }
            jan. 5 -> date { month: "january" day: "five" }
            5 january 2012 -> date { day: "five" month: "january" year: "twenty twelve" }
            2012-01-05 -> date { year: "twenty twelve" month: "january" day: "five" }
            2012.01.05 -> date { year: "twenty twelve" month: "january" day: "five" }
            2012/01/05 -> date { year: "twenty twelve" month: "january" day: "five" }
            2012 -> date { year: "twenty twelve" }
        """
        cardinal = Cardinal(self.deterministic)
        # january, January, JANUARY
        month_graph = string_file(get_abs_path("english/data/date/month_name.tsv"))
        # jan, Jan, JAN
        month_abbr_graph = string_file(get_abs_path("english/data/date/month_abbr.tsv"))
        month_graph |= month_abbr_graph
        month_graph += delete(self.PUNCT).ques

        month_numbers_labels = string_file(
            get_abs_path("english/data/date/month_number.tsv")
        )
        cardinal_graph = cardinal.graph_hundred_component_at_least_one_none_zero_digit

        year_graph = _get_year_graph(
            cardinal_graph=cardinal_graph, deterministic=self.deterministic
        )

        # three_digit_year = (self.DIGIT @ cardinal_graph) + insert(" ") + (self.DIGIT ** 2) @ cardinal_graph
        # year_graph |= three_digit_year

        month_graph = insert('month: "') + month_graph + insert('"')
        month_numbers_graph = insert('month: "') + month_numbers_labels + insert('"')

        endings = ["rd", "th", "st", "nd"]
        endings += [x.upper() for x in endings]
        endings = union(*endings)

        day_graph = (
            insert('day: "')
            + delete("the ").ques
            + (
                ((union("1", "2") + self.DIGIT) | self.DIGIT | union("30", "31"))
                + delete(endings).ques
            )
            @ cardinal_graph
            + insert('"')
        )

        two_digit_year = _get_two_digit_year(
            cardinal_graph=cardinal_graph,
            single_digits=cardinal.single_digits,
        )
        two_digit_year = (
            insert('year: "') + two_digit_year + union(",", ".").ques + insert('"')
        )

        graph_year = (
            accep(" ")
            + insert('year: "')
            + year_graph
            + union(",", ".").ques
            + insert('"')
        )
        graph_year |= (
            insert(' year: "')
            + accep(",")
            + accep(" ").ques
            + year_graph
            + union(",", ".").ques
            + insert('"')
        )
        optional_graph_year = graph_year.ques

        year_graph = insert('year: "') + year_graph + insert('"')

        graph_mdy = month_graph + (
            (self.DELETE_EXTRA_SPACE + day_graph)
            | (accep(" ") + day_graph)
            | graph_year
            | (self.DELETE_EXTRA_SPACE + day_graph + graph_year)
        )
        graph_mdy |= (
            month_graph
            + cross("-", " ")
            + day_graph
            + ((cross("-", " ") + self.VSIGMA) @ graph_year).ques
        )
        for x in ["-", "/", "."]:
            delete_sep = delete(x)
            graph_mdy |= (
                month_numbers_graph
                + delete_sep
                + insert(" ")
                + delete("0").ques
                + day_graph
                + delete_sep
                + insert(" ")
                + add_weight(year_graph, -1.0)
            )

        graph_dmy = (
            day_graph
            + self.DELETE_EXTRA_SPACE
            + insert(" ")
            + month_graph
            + optional_graph_year
        )
        day_ex_month = (
            self.DIGIT**2 - project(month_numbers_graph, "input")
        ) @ day_graph
        for x in ["-", "/", "."]:
            delete_sep = delete(x)
            graph_dmy |= (
                day_ex_month
                + delete_sep
                + insert(" ")
                + month_numbers_graph
                + delete_sep
                + insert(" ")
                + (add_weight(year_graph, -1.0))
            )

        graph_ymd = (
            year_graph
            + self.DELETE_EXTRA_SPACE
            + insert(" ")
            + month_graph
            + self.DELETE_EXTRA_SPACE
            + insert(" ")
            + day_graph
        )
        for x in ["-", "/", "."]:
            delete_sep = delete(x)
            graph_ymd |= (
                (add_weight(year_graph, -1.0))
                + delete_sep
                + insert(" ")
                + month_numbers_graph
                + delete_sep
                + insert(" ")
                + delete("0").ques
                + day_graph
            )

        final_graph = add_weight(graph_mdy | graph_dmy | graph_ymd, -0.1) | year_graph

        period_fy = insert('text: "') + _get_financial_period_graph() + insert('"')
        graph_fy = period_fy + insert(" ") + two_digit_year

        final_graph |= graph_fy

        self.tagger = self.add_tokens(final_graph)

    def build_verbalizer(self):
        """
        Finite state transducer for verbalizing date, e.g.
            date { month: "february" day: "five" year: "twenty twelve" } -> the fifth of february twenty twelve
            date { day: "five" month: "february" year: "twenty twelve" } -> the fifth of february twenty twelve
        """
        ordinal = Ordinal(self.deterministic)
        phrase = self.NOT_QUOTE.plus
        day_cardinal = (
            delete("day:") + self.DELETE_SPACE + delete('"') + phrase + delete('"')
        )
        day = day_cardinal @ ordinal.suffix
        period = (
            delete("text:") + self.DELETE_SPACE + delete('"') + phrase + delete('"')
        )
        month = (
            delete("month:") + self.DELETE_SPACE + delete('"') + phrase + delete('"')
        )

        year = (
            delete("year:")
            + self.DELETE_SPACE
            + delete('"')
            + phrase
            + self.DELETE_SPACE
            + delete('"')
        )

        # financial period
        graph_fy = (
            insert("the ")
            + period
            + insert(" of")
            + (self.DELETE_EXTRA_SPACE + year).ques
        )

        # day month year, month year
        graph_dmy = (
            (insert("the ") + day + self.DELETE_EXTRA_SPACE + insert("of ")).ques
            + month
            + (self.DELETE_EXTRA_SPACE + year).ques
        )

        final_graph = (graph_dmy | year | graph_fy) + self.DELETE_SPACE
        self.verbalizer = self.delete_tokens(final_graph)
