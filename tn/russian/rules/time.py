# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
from pynini.lib import pynutil

from tn.processor import Processor
from tn.russian.rules.utils import DIGIT, data_path, number_names
from tn.utils import load_labels


# Each reading of "HH:MM" names its fields; the parser puts minutes first for
# "семь минут девятого" (past_*) and "без пятнадцати минут шесть" (to_*).
TOKEN_ORDER = {"time": ["hours", "minutes", "past_minutes", "past_hour", "to_minutes", "to_hour"]}


class Time(Processor):
    """Clock times, e.g. "13:05" -> time { hours: "13" minutes: "05" }."""

    def __init__(self):
        super().__init__("time", token_orders=TOKEN_ORDER)
        number = pynini.closure(pynini.cross("0", ""), 0, 1) + number_names().nominative
        increment_ordinal = pynini.string_file(data_path("time/increment_hour_ordinal.tsv"))
        increment_cardinal = pynini.string_file(data_path("time/increment_hour_cardinal.tsv"))
        minutes_to_hour = pynini.string_file(data_path("time/minutes_to_hour.tsv"))

        hour_options = pynini.union(
            *[row[0] for row in load_labels(data_path("time/increment_hour_ordinal.tsv"))],
            *[row[1] for row in load_labels(data_path("time/time_convert.tsv"))],
        ).optimize()
        hour_one = pynini.union("01", "21")
        hour_few = pynini.union("02", "03", "04", "22", "23")
        hour_many = pynini.difference(hour_options, hour_one | hour_few)
        hours = (
            hour_one @ number + pynutil.insert(" час")
            | hour_few @ number + pynutil.insert(" часа")
            | hour_many @ number + pynutil.insert(" часов")
        ).optimize()

        start = pynini.union(*"012345")
        minute_one = start + "1"
        minute_few = pynini.difference(start + pynini.union(*"234"), pynini.union("12", "13", "14"))
        minute_many = pynini.difference(start + DIGIT, minute_one | minute_few)
        minutes = (
            minute_one @ number + pynutil.insert(" минута")
            | minute_few @ number + pynutil.insert(" минуты")
            | minute_many @ number + pynutil.insert(" минут")
        ).optimize()
        next_hour = (hour_options @ increment_ordinal).optimize()
        hour_to = (hour_options @ increment_cardinal).optimize()

        colon = pynutil.delete(":") + pynutil.insert(" ")
        self.tagger = self.add_tokens(
            self.tag_field("hours", hours) + colon + self.tag_field("minutes", minutes)
            | self.tag_field("past_hour", next_hour) + colon + self.tag_field("past_minutes", minutes)
            | self.tag_field("to_hour", hour_to) + colon + self.tag_field("to_minutes", minutes_to_hour)
        )

        optional_and = pynini.closure(pynutil.insert("и "), 0, 1)
        clock = self.verbalize_field("hours", hours) + (
            pynini.accep(" ") + self.verbalize_field("minutes", optional_and + minutes)
            | pynutil.delete(" ") + self.verbalize_field("minutes", pynutil.add_weight(pynutil.delete("00"), -0.5))
        )
        past = self.verbalize_field("past_minutes", minutes) + pynini.accep(" ")
        past += self.verbalize_field("past_hour", next_hour)
        to = self.verbalize_field("to_minutes", pynutil.insert("без ") + minutes_to_hour) + pynini.accep(" ")
        to += self.verbalize_field("to_hour", hour_to)
        self.verbalizer = self.delete_tokens(clock | past | to)
