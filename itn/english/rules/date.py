# Copyright (c) 2026 Zhendong Peng (pzd17@tsinghua.org.cn)
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
from pynini import closure, cross, string_file
from pynini.lib.pynutil import add_weight, delete, insert

from itn.english.rules.cardinal import Cardinal
from itn.english.rules.ordinal import Ordinal
from tn.processor import Processor
from tn.utils import get_abs_path


class Date(Processor):

    def __init__(self, cardinal=None, ordinal=None):
        super().__init__(name="date", ordertype="itn")
        self.cardinal = cardinal or Cardinal()
        self.ordinal = ordinal or Ordinal(cardinal=self.cardinal)
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        ds = delete(" ")
        month_names = string_file(get_abs_path("../itn/english/data/months.tsv"))
        month = insert('month: "') + month_names + insert('"')

        # Day: accept ordinal words ("fifth", "twenty first") or cardinal
        # words ("thirty") -- both resolve to a number via the cardinal graph.
        # Restrict to 1-31 range via composition with DIGIT{1,2}.
        day_graph = self.ordinal.graph | self.cardinal.graph
        day_graph = pynini.compose(day_graph, self.DIGIT + closure(self.DIGIT, 0, 1))
        day = insert('day: "') + day_graph + insert('"')

        # Year graph: handles common spoken year forms
        digit = string_file(get_abs_path("../itn/english/data/numbers/digit.tsv"))
        teen = string_file(get_abs_path("../itn/english/data/numbers/teen.tsv"))
        ties = string_file(get_abs_path("../itn/english/data/numbers/ties.tsv"))

        # Two-digit: 10-99
        two_digit = teen | (ties + (ds + digit | insert("0")))

        # "oh five" / "o five" => 05
        oh_digit = (cross("oh", "0") | cross("o", "0")) + ds + digit

        # Year as two groups of two digits: "twenty twelve" => 2012
        year_two_parts = (teen | two_digit) + ds + (two_digit | oh_digit | teen)

        # Year as "X thousand Y": "two thousand twelve" => 2012
        # Need zero-padded variants so "two thousand three" => 2003
        hundreds = digit + ds + delete("hundred") + (ds + two_digit | ds + insert("0") + digit | insert("00"))
        up_to_999_padded = hundreds | insert("0") + two_digit | insert("00") + digit
        year_thousands = (
            digit
            + ds
            + delete("thousand")
            + (ds + up_to_999_padded | insert("000"))
        )

        # Year as hundreds: "nineteen oh five" => 1905
        year_hundreds = (teen | two_digit) + ds + oh_digit

        year_graph = year_two_parts | year_thousands | year_hundreds

        # Delete optional "and" within year
        delete_and = self.build_rule(delete("and "), " ", self.ALPHA)
        year_graph = (delete_and @ year_graph).optimize()

        year = insert('year: "') + year_graph + insert('"')

        # Marker to preserve field order through TokenParser
        po = insert(' preserve_order: "true"')

        # Format: month day year => "july twenty fifth two thousand twelve"
        graph_mdy = month + ds + insert(" ") + day + ds + insert(" ") + year + po
        # Format: month day (no year) => "january first"
        graph_md = month + ds + insert(" ") + day + po
        # Format: month year (no day) => "july two thousand twelve"
        graph_my = month + ds + insert(" ") + add_weight(year, -0.1) + po
        # Format: "the day of month year" => "the twenty fifth of july twenty twelve"
        graph_dmy = (
            delete("the")
            + ds
            + day
            + ds
            + delete("of")
            + ds
            + insert(" ")
            + month
            + ds
            + insert(" ")
            + year
            + po
        )
        # Format: "the day of month" (no year) => "the fifteenth of january"
        graph_dm = (
            delete("the")
            + ds
            + day
            + ds
            + delete("of")
            + ds
            + insert(" ")
            + month
            + po
        )
        # Year only => "twenty twelve", "two thousand three"
        graph_y = add_weight(year, 0.01) + po

        # Decades: "nineteen eighties" => 1980s
        decade_suffix = closure(self.ALPHA, 1) + (cross("ies", "y") | delete("s"))
        decade_word = pynini.compose(decade_suffix, ties | cross("ten", "10"))
        graph_decade = (
            insert('year: "') + (teen | two_digit) + ds + decade_word + insert('0s"') + po
        )

        final_graph = graph_mdy | graph_md | graph_my | graph_dmy | graph_dm | graph_y | graph_decade
        self.tagger = self.add_tokens(final_graph)

    def build_verbalizer(self):
        month = (
            delete("month:")
            + self.DELETE_SPACE
            + delete('"')
            + self.NOT_QUOTE.plus
            + delete('"')
        )
        day = (
            delete("day:")
            + self.DELETE_SPACE
            + delete('"')
            + self.NOT_QUOTE.plus
            + delete('"')
        )
        year = (
            delete("year:")
            + self.DELETE_SPACE
            + delete('"')
            + self.NOT_QUOTE.plus
            + delete('"')
        )
        delete_po = (
            delete("preserve_order:")
            + self.DELETE_SPACE
            + delete('"')
            + delete("true")
            + delete('"')
        )

        optional_day = closure(self.DELETE_SPACE + insert(" ") + day, 0, 1)
        optional_year = closure(self.DELETE_SPACE + insert(" ") + year, 0, 1)

        # month (day) (year)
        graph_mdy = month + optional_day + optional_year
        # day month (year)
        graph_dmy = day + self.DELETE_SPACE + insert(" ") + month + optional_year
        # year only
        graph_y = year

        graph = (graph_mdy | graph_dmy | graph_y) + self.DELETE_SPACE + delete_po
        self.verbalizer = self.delete_tokens(graph)
