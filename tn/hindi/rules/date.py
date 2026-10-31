# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from pynini.lib.pynutil import add_weight, delete, insert

from tn.hindi.rules.utils import ALL_DIGIT, ALL_NON_ZERO, ALL_ZERO, hindi_path, read_lines
from tn.processor import Processor

TOKEN_ORDER = {}


class Date(Processor):
    """Dates keep their written day/month/year (or era phrase) raw.

    ``25-03-2020`` -> ``date { day: "25" month: "03" year: "2020" }``;
    ``1999 में`` -> ``date { era: "1999 में" }``.
    """

    def __init__(self, cardinal):
        super().__init__("date", token_orders=TOKEN_ORDER)
        days = pynini.string_file(hindi_path("date/days.tsv"))
        months = pynini.string_file(hindi_path("date/months.tsv"))
        year_suffix = pynini.string_file(hindi_path("date/year_suffix.tsv"))
        # NeMo projects its module-level digit map in place while building the spoken-year
        # acceptor, so every later use of "digit" in its date grammar accepts the spoken
        # words (एक..नौ) instead of digits. That behaviour is kept for parity.
        digit = cardinal.digit.copy().project("output").optimize()
        teens_ties_hi = pynini.string_file(hindi_path("numbers/teens_and_ties.tsv"))
        teens_and_ties = cardinal.teens_and_ties
        suffixes = pynini.string_map(read_lines("date/suffixes.tsv"))
        prefixes = pynini.string_map(read_lines("date/prefixes.tsv"))

        hundreds_word = teens_ties_hi.copy().project("output")
        unit_word = pynini.union(hundreds_word, digit)
        spoken_year = hundreds_word + pynini.accep(" सौ") + (pynini.accep(" ") + unit_word).ques

        pad_latin = pynini.union(*[pynini.cross(str(i), "0{}".format(i)) for i in range(1, 10)])
        pad_devanagari = pynini.union(*[pynini.cross(d, "०" + d) for d in "१२३४५६७८९"])
        pad = pad_latin | pad_devanagari

        year_thousands = (ALL_DIGIT + ALL_ZERO + ALL_DIGIT + ALL_DIGIT) @ cardinal.graph_thousands
        year_hundreds = (ALL_DIGIT + ALL_NON_ZERO + ALL_DIGIT + ALL_DIGIT) @ cardinal.graph_hundreds_as_thousand
        year = pynini.union(year_thousands, year_hundreds).optimize()
        year_era = pynini.union(year_thousands, year_hundreds, cardinal.graph_hundreds).optimize()
        number = pynini.union(digit, teens_and_ties, cardinal.graph_hundreds, year_thousands, year_hundreds)

        day_padded = pynini.union(days, teens_and_ties).optimize()
        day_bare = (pad @ days).optimize()
        month_name = months.copy().project("output").optimize()
        month_padded = pynini.union(months, month_name).optimize()
        month_bare = pynini.union(pad @ months, month_name).optimize()

        dash = delete("-") + insert(" ")
        comma = delete(",") + delete(" ").ques + insert(" ")
        space = delete(" ") + insert(" ")

        # Era phrases are single raw fields whose whole reading depends on the pattern.
        eras = [
            (year_era + insert(" ") + year_suffix, -0.005),
            (number + insert(" ") + pynini.cross("-", "से") + insert(" ") + number, -0.005),
            (year_suffix, -0.001),
            (((ALL_DIGIT.plus @ number) + pynini.accep("वीं")), -0.001),
            (prefixes + pynini.accep(" ") + spoken_year + suffixes, -0.012),
            (prefixes + pynini.accep(" ") + spoken_year, -0.011),
            (spoken_year + suffixes, -0.010),
            (spoken_year, -0.009),
            (prefixes + pynini.accep(" ") + year + suffixes, -0.010),
            (prefixes + pynini.accep(" ") + year, -0.009),
            (year + suffixes, -0.001),
        ]
        era = pynini.union(*[add_weight(graph, weight) for graph, weight in eras]).optimize()

        def day(graph):
            return self.tag_field("day", graph), self.verbalize_field("day", graph)

        def month(graph):
            return self.tag_field("month", graph), insert(" ") + self.verbalize_field("month", graph, leading_space=True)

        year_tag = self.tag_field("year", year)
        year_verb = insert(" ") + self.verbalize_field("year", year, leading_space=True)
        era_tag = self.tag_field("era", era)
        day_p, day_pv = day(day_padded)
        day_b, day_bv = day(day_bare)
        month_p, month_pv = month(month_padded)
        month_b, month_bv = month(month_bare)
        month_n, month_nv = month(months)
        month_name_tag = self.tag_field("month", month_name)
        month_name_verb = self.verbalize_field("month", month_name)
        month_first_tag = self.tag_field("month", month_padded)
        month_first_verb = self.verbalize_field("month", month_padded)
        # NeMo only attaches an era to a full date when it is glued to the year ("१९९०ई.").
        era_glued = insert(" ") + self.tag_field("era", year_suffix)
        era_verb = self.verbalize_field("era", era)
        era_after = insert(" ") + self.verbalize_field("era", year_suffix, leading_space=True)

        taggers = [
            (day_p + space + month_p + comma + year_tag + era_glued, -0.003),
            (month_first_tag + comma + year_tag + era_glued, -0.003),
            (day_p + dash + month_p + dash + year_tag, -0.001),
            (day_b + dash + month_b + dash + year_tag, -0.001),
            (day_p + space + month_p + comma + year_tag, -0.001),
            (day_p + dash + month_p, -0.001),
            (day_b + dash + month_b, -0.001),
            (day_p + space + month_n, -0.001),
            (month_name_tag + space + year_tag, -0.2),
            (month_first_tag + comma + year_tag, -0.2),
            (era_tag, 0),
        ]
        self.tagger = self.add_tokens(pynini.union(*[add_weight(graph, weight) for graph, weight in taggers]))

        day_v = day_pv | day_bv
        month_v = month_pv | month_bv | month_nv
        month_lead = month_first_verb | month_name_verb
        verbalizers = [
            day_v + month_v,
            day_v + month_v + year_verb,
            day_v + month_v + year_verb + era_after,
            month_lead + year_verb,
            month_lead + year_verb + era_after,
            era_verb,
        ]
        self.verbalizer = self.delete_tokens(pynini.union(*verbalizers))
