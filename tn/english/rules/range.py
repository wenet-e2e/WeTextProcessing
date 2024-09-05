# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from pynini import accep, cross, string_file, union
from pynini.lib.pynutil import delete, insert

from tn.english.rules.cardinal import Cardinal
from tn.english.rules.time import Time
from tn.english.rules.date import Date
from tn.processor import Processor
from tn.utils import get_abs_path


class Range(Processor):

    def __init__(self, deterministic: bool = False):
        """
        Args:
            deterministic: if True will provide a single transduction option,
                for False multiple transduction are generated (used for audio-based normalization)
        """
        super().__init__("range", ordertype="en_tn")
        self.deterministic = deterministic
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        """
        Finite state transducer for verbalizing range, e.g.
            2-3 => range { value "two to three" }
        """
        cardinal = Cardinal(self.deterministic).graph_with_and
        time = Time(self.deterministic)
        time = time.tagger @ time.verbalizer
        date = Date(self.deterministic)
        date = date.tagger @ date.verbalizer
        week = string_file(get_abs_path("english/data/date/week.tsv"))
        delete_space = delete(" ").ques

        approx = cross("~", "approximately")

        # WEEK
        week_graph = (
            week + delete_space + (cross("-", " to ") | approx) + delete_space + week
        )

        # TIME
        time_graph = time + delete_space + cross("-", " to ") + delete_space + time
        self.graph = time_graph | (approx + time) | week_graph

        # YEAR
        date_year_four_digit = (self.DIGIT**4 + accep("s").ques) @ date
        date_year_two_digit = (self.DIGIT**2 + accep("s").ques) @ date
        year_to_year_graph = (
            date_year_four_digit
            + delete_space
            + cross("-", " to ")
            + delete_space
            + (date_year_four_digit | date_year_two_digit | (self.DIGIT**2 @ cardinal))
        )
        mid_year_graph = (
            accep("mid")
            + cross("-", " ")
            + (date_year_four_digit | date_year_two_digit)
        )

        self.graph |= year_to_year_graph
        self.graph |= mid_year_graph

        # ADDITION
        range_graph = cardinal + (cross("+", " plus ") + cardinal).plus
        range_graph |= cardinal + (cross(" + ", " plus ") + cardinal).plus
        range_graph |= approx + cardinal
        range_graph |= cardinal + (cross("...", " ... ") | accep(" ... ")) + cardinal

        if not self.deterministic:
            # cardinal ----
            cardinal_to_cardinal_graph = (
                cardinal
                + delete_space
                + cross("-", union(" to ", " minus "))
                + delete_space
                + cardinal
            )

            range_graph |= cardinal_to_cardinal_graph | (
                cardinal + delete_space + cross(":", " to ") + delete_space + cardinal
            )

            # MULTIPLY
            for x in [" x ", "x"]:
                range_graph |= cardinal + cross(x, union(" by ", " times ")) + cardinal

            # 40x -> "40 times" ("40 x" cases is covered in serial)
            for x in [" x", "x"]:
                range_graph |= cardinal + cross(x, " times")

                # 5x to 7x-> five to seven x/times
                range_graph |= (
                    cardinal
                    + delete(x)
                    + union(" to ", "-", " - ")
                    + cardinal
                    + cross(x, union(" x", " times"))
                )

            for x in ["*", " * "]:
                range_graph |= cardinal + (cross(x, " times ") + cardinal).plus

            # supports "No. 12" -> "Number 12"
            range_graph |= (
                (cross(union("NO", "No"), "Number") | cross("no", "number"))
                + union(". ", " ").ques
                + cardinal
            )

            for x in ["/", " / "]:
                range_graph |= cardinal + (cross(x, " divided by ") + cardinal).plus

            # 10% to 20% -> ten to twenty percent
            range_graph |= (
                cardinal
                + (cross("%", " percent") | delete("%")).ques  # noqa
                + union(" to ", "-", " - ")
                + cardinal  # noqa
                + cross("%", " percent")
            )  # noqa

        self.graph |= range_graph

        final_graph = insert('value: "') + self.graph + insert('"')
        self.tagger = self.add_tokens(final_graph)
