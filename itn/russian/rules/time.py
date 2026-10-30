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

from itn.russian.rules.utils import DIGIT, data_path, number_names
from itn.utils import load_labels
from tn.processor import Processor


# Each reading names its fields; the parser puts the hour first for
# "семь минут девятого" (past_*) and "без пятнадцати минут шесть" (to_*).
TOKEN_ORDER = {"time": ["hours", "minutes", "past_hour", "past_minutes", "to_hour", "to_minutes"]}


class Time(Processor):
    """Spoken clock times, e.g. "два часа и три минуты" -> time { hours: "два часа" minutes: "и три минуты" }."""

    def __init__(self):
        super().__init__(name="time", ordertype="itn")
        number = pynini.closure(pynini.cross("0", ""), 0, 1) + number_names().nominative
        increment_ordinal = pynini.string_file(data_path("time/increment_hour_ordinal.tsv"))
        increment_cardinal = pynini.string_file(data_path("time/increment_hour_cardinal.tsv"))
        minutes_to_hour = pynini.string_file(data_path("time/minutes_to_hour.tsv"))
        two_digit_hours = pynini.union(*[row[0] for row in load_labels(data_path("time/increment_hour_ordinal.tsv"))])
        other_hours = pynini.union(*[row[1] for row in load_labels(data_path("time/time_convert.tsv"))])
        # "пять часов" is written "05" rather than "5".
        hour_options = (
            two_digit_hours | pynutil.add_weight(pynini.difference(other_hours, two_digit_hours), 0.1)
        ).optimize()

        # Written -> spoken readings as in TN, inverted below.
        hour_one = pynini.union("01", "21")
        hour_few = pynini.union("02", "03", "04", "22", "23")
        hour_many = pynini.difference(hour_options, hour_one | hour_few)
        hours = (
            hour_one @ number + pynutil.insert(" час")
            | hour_few @ number + pynutil.insert(" часа")
            | hour_many @ number + pynutil.insert(" часов")
        )
        start = pynini.union(*"012345")
        minute_one = start + "1"
        minute_few = pynini.difference(start + pynini.union(*"234"), pynini.union("12", "13", "14"))
        minute_many = pynini.difference(start + DIGIT, minute_one | minute_few)
        minutes = (
            minute_one @ number + pynutil.insert(" минута")
            | minute_few @ number + pynutil.insert(" минуты")
            | minute_many @ number + pynutil.insert(" минут")
        )
        self.hours = hours.invert().optimize()
        self.minutes = (pynini.closure(pynutil.insert("и "), 0, 1) + minutes).invert().optimize()
        self.past_minutes = minutes.invert().optimize()
        self.past_hour = (hour_options @ increment_ordinal).invert().optimize()
        self.to_minutes = (pynutil.insert("без ") + minutes_to_hour).invert().optimize()
        self.to_hour = (hour_options @ increment_cardinal).invert().optimize()

        space = pynutil.delete(" ") + pynutil.insert(" ")
        self.tagger = self.add_tokens(
            self.tag_field("hours", self.hours) + (space + self.tag_field("minutes", self.minutes)).ques
            | self.tag_field("past_minutes", self.past_minutes) + space + self.tag_field("past_hour", self.past_hour)
            | self.tag_field("to_minutes", self.to_minutes) + space + self.tag_field("to_hour", self.to_hour)
        )
        colon = pynutil.delete(" ") + pynutil.insert(":")
        self.verbalizer = self.delete_tokens(
            self.verbalize_field("hours", self.hours)
            + (colon + self.verbalize_field("minutes", self.minutes) | pynutil.insert(":00"))
            | self.verbalize_field("past_hour", self.past_hour) + colon
            + self.verbalize_field("past_minutes", self.past_minutes)
            | self.verbalize_field("to_hour", self.to_hour) + colon + self.verbalize_field("to_minutes", self.to_minutes)
        )
