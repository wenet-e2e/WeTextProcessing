# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pynini
from pynini.lib import pynutil

from tn.processor import Processor
from tn.utils import get_abs_path, load_labels


TOKEN_ORDER = {"date": ["day", "month", "year"]}


class Date(Processor):
    """German calendar tokens retain written components, including zero padding."""

    def __init__(self, cardinal, ordinal, deterministic=True):
        super().__init__("date", token_orders=TOKEN_ORDER)
        self.deterministic = deterministic
        strip_zero = (pynutil.delete("0") | (self.DIGIT - "0")) + self.DIGIT
        component = strip_zero | self.DIGIT
        ordinal_ending = pynini.cdrewrite(ordinal.stem, "", "[EOS]", self.VSIGMA) + pynutil.insert("ter")
        day_number = component @ pynini.union(*[str(day) for day in range(1, 32)])
        month_number = component @ pynini.union(*[str(month) for month in range(1, 13)])
        self.day_graph = (day_number @ cardinal.hundreds @ ordinal_ending).optimize()

        labels = load_labels(get_abs_path("german/data/months/abbr_to_name.tsv"))
        abbreviations = pynini.string_map(labels)
        lower_initial = self.TO_LOWER + pynini.closure(self.VCHAR)
        self.month_abbr_graph = (
            (pynutil.add_weight(abbreviations, 0.0001) | lower_initial @ abbreviations)
            + pynini.closure(pynutil.delete(".", weight=-0.0001), 0, 1)
        ).optimize()
        names = pynini.union(*[name for _, name in labels])
        self.named_month_graph = (names | lower_initial @ names | self.month_abbr_graph).optimize()
        month_names = month_number @ pynini.string_file(get_abs_path("german/data/months/numbers.tsv"))
        month_ordinals = month_number @ cardinal.hundreds @ ordinal_ending
        self.numeric_month_graph = (month_names | pynutil.add_weight(month_ordinals, 0.0001)).optimize()
        self.month_graph = (self.named_month_graph | self.numeric_month_graph).optimize()

        # Bare year semantics: 10xx--19xx in paired hundreds, 20xx also in
        # paired tens, and 20xx--21xx as ordinary cardinals. No token labels.
        two_digits = strip_zero @ cardinal.two_digits
        first_pair = (pynini.accep("1") + self.DIGIT) @ two_digits
        hundred = pynutil.insert("hundert")
        paired = (
            first_pair + pynutil.insert(" ")
            + pynini.closure(hundred + pynutil.insert(" "), 0, 1) + two_digits
        )
        paired_twenty = pynini.accep("20") @ two_digits + pynutil.insert(" ") + two_digits
        centuries = first_pair + pynutil.insert(" ") + pynutil.delete("00") + hundred
        recent = (pynini.union("20", "21") + pynini.closure(self.DIGIT, 2, 2)) @ cardinal.graph
        self.year_graph = pynutil.add_weight(paired | paired_twenty | centuries | recent, 0.001).optimize()
        # Raw-field projection collapses the source's equal-cost spoken tags.
        # Resolve that tie in V to retain its ordinary top1, without changing
        # the public source year graph or dropping the paired audio candidate.
        year_verbalization = pynutil.add_weight(
            paired | pynutil.add_weight(paired_twenty, 0.000001) | centuries | recent, 0.001,
        ).optimize()

        day = self.tag_field("day", self.day_graph)
        month_name = self.tag_field("month", self.named_month_graph)
        month_numeric = self.tag_field("month", self.numeric_month_graph)
        year = self.tag_field("year", self.year_graph)
        named_date = (
            day + pynutil.delete(".") + pynini.closure(pynutil.delete(" "), 0, 1)
            + pynutil.insert(" ") + month_name
            + pynini.closure(pynini.accep(" ") + year, 0, 1)
        )
        numeric_date = (
            day + pynini.cross(".", " ") + month_numeric
            + pynini.closure(pynini.cross(".", " ") + year, 0, 1)
        )
        iso_date = (
            year + pynini.cross("-", " ") + month_numeric
            + pynini.closure(pynini.cross("-", " ") + day, 0, 1)
        )
        self.fields_tagger = (named_date | numeric_date | iso_date | year).optimize()
        self.tagger = self.add_tokens(self.fields_tagger)

        day_v = self.verbalize_field("day", self.day_graph)
        month_v = self.verbalize_field("month", self.month_graph)
        year_v = self.verbalize_field("year", year_verbalization)
        self.fields_verbalizer = (
            day_v + pynini.accep(" ") + month_v
            + pynini.closure(pynini.accep(" ") + year_v, 0, 1)
            | month_v + pynini.accep(" ") + year_v
            | year_v
        ).optimize()
        self.verbalizer = self.delete_tokens(self.fields_verbalizer)
