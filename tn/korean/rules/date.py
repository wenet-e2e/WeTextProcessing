# Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
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

from tn.processor import Processor
from tn.utils import get_abs_path


TOKEN_ORDER = {"date": ["era", "year", "month", "day", "weekday", "range_start"]}


class Date(Processor):
    """Korean calendar fields, with all numeral conversion in the verbalizer."""

    def __init__(self, cardinal):
        super().__init__("date", token_orders=TOKEN_ORDER)
        strip_zero = delete("0").ques
        month_inputs = pynini.union(*[str(i) for i in range(1, 13)])
        day_inputs = pynini.union(*[str(i) for i in range(1, 32)])
        exceptions = pynini.string_file(get_abs_path("korean/data/date/exceptions.tsv"))
        regular_month = pynini.difference(month_inputs, self.input_projection(exceptions)) @ cardinal.graph
        month_number = (strip_zero + (exceptions | regular_month)).optimize()
        day_number = (strip_zero + (day_inputs @ cardinal.graph)).optimize()
        year_digits = (self.DIGIT - "0") + pynini.closure(self.DIGIT, 0, 3)
        year_number = (delete("0").closure(0, 3) + (year_digits @ cardinal.graph)).optimize()
        year_four = ((pynini.union("1", "2") + self.DIGIT**3) @ cardinal.graph).optimize()
        year_two = (self.DIGIT**2 @ cardinal.graph).optimize()
        era = pynini.union("기원전", "기원후")

        week = pynini.string_file(get_abs_path("korean/data/date/week.tsv"))
        full_week = week.copy().project("output").optimize()
        weekday = (week | full_week | week + pynini.cross("〜", "부터") + week
                   | week + delete("・") + week).optimize()
        brackets = []
        for left, right in [("(", ")"), ("（", "）")]:
            brackets.append(self.DELETE_SPACE + delete(left) + self.DELETE_SPACE
                            + self.tag_field("weekday", weekday)
                            + self.DELETE_SPACE + delete(right) + self.DELETE_SPACE)
        weekday_tag = self.tag_field("weekday", week | full_week) | pynini.union(*brackets)

        month_suffix = pynini.accep("월") + pynini.union("에", "은", "는", "에는").ques
        written = {
            "year": year_number + pynini.union("년", "년대"),
            "month": month_number + month_suffix,
            "day": day_number + "일",
            "weekday": weekday,
            "era": era,
            "range_start": day_number,
        }
        # Separated dates omit only their structural separators; component values
        # retain leading zeros. Literal Korean suffixes remain inside raw fields.
        semantic = {
            "year": (year_number + pynini.union("년", "년대") | year_four + insert("년")
                     | year_two + insert("년")),
            "month": month_number + month_suffix | month_number + insert("월"),
            "day": day_number + "일" | day_number + insert("일"),
            "weekday": weekday,
            "era": era,
            "range_start": day_number,
        }
        tags = {key: self.tag_field(key, graph) for key, graph in written.items()}
        tags["weekday"] = weekday_tag
        forms = []
        sequences = [("year",), ("month",), ("day",), ("weekday",),
                     ("year", "month"), ("month", "day"), ("year", "month", "day"),
                     ("month", "range_start")]
        for fields in sequences:
            graph = tags[fields[0]].copy()
            for field in fields[1:]:
                graph += self.DELETE_SPACE + insert(" ") + tags[field]
            if fields[0] == "year":
                graph = (tags["era"] + self.DELETE_SPACE + insert(" ")).ques + graph
            if fields[-1] not in ("weekday", "range_start"):
                graph += (insert(" ") + self.DELETE_SPACE + weekday_tag).ques
            forms.append(graph)

        # Month/day versus day/month is classified in written order. TokenParser
        # alone performs the unbounded exchange into year/month/day reading order.
        for separator in ["/", "-", "."]:
            sep = delete(separator) + self.DELETE_SPACE + insert(" ")
            year = self.tag_field("year", year_four)
            month = self.tag_field("month", month_number)
            day = self.tag_field("day", day_number)
            numeric_date = (tags["era"] + self.DELETE_SPACE + insert(" ")).ques
            numeric_date += year + sep + month + sep + day
            optional_week = (insert(" ") + self.DELETE_SPACE + weekday_tag).ques
            forms.append(numeric_date + optional_week)
            if separator == ".":
                forms.append(numeric_date + delete(".") + insert(" ") + weekday_tag)
            strict_month = (self.DIGIT**2 @ month_number).optimize()
            strict_day = (self.DIGIT**2 @ day_number).optimize()
            end_year = self.tag_field("year", year_four | add_weight(year_two, 1.0))
            american = self.tag_field("month", strict_month) + sep
            american += self.tag_field("day", strict_day) + sep + end_year
            european = self.tag_field("day", strict_day) + sep
            european += self.tag_field("month", strict_month) + sep + end_year
            forms.append(add_weight(american, -0.0001) + optional_week)
            forms.append(european + optional_week)
        self.tagger = self.add_tokens(pynini.union(*forms))

        readings = []
        for fields in sequences:
            graph = self.verbalize_field(fields[0], semantic[fields[0]])
            for field in fields[1:]:
                graph += insert(" ") + self.verbalize_field(field, semantic[field], leading_space=True)
            if fields[0] == "year":
                era_reading = self.verbalize_field("era", era) + insert(" ") + delete(" ")
                graph = era_reading.ques + graph
            if fields[-1] not in ("weekday", "range_start"):
                with_week = graph + insert(" ")
                with_week += self.verbalize_field("weekday", weekday, leading_space=True)
                readings.append(with_week)
            if fields[-1] in ("year", "month"):
                graph += insert(" ")
            readings.append(graph)
        self.verbalizer = self.delete_tokens(pynini.union(*readings))
