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
from tn.utils import get_abs_path


TOKEN_ORDER = {"time": ["kind", "hours", "minutes", "seconds", "zone"]}


class Time(Processor):
    def __init__(self, tn_time):
        super().__init__("time", ordertype="itn", token_orders=TOKEN_ORDER)
        values = pynini.union(*[str(value) for value in range(100)])
        numbers = (values @ tn_time.number_graph).optimize()
        omit_spaces = pynini.closure(self.VSIGMA | pynutil.delete(" ", weight=0.0001))

        def inverse(graph):
            return (graph @ omit_spaces).invert().optimize()

        self.number_graph = inverse(numbers)
        hour_numbers = numbers @ self.build_rule(pynini.cross("eins", "ein"), l="[BOS]", r="[EOS]")
        self.hour_graph = inverse(hour_numbers)
        self.zone_graph = inverse(tn_time.zone_graph)
        separator = pynini.closure(pynutil.delete(" ", weight=0.0001), 0, 1) + pynutil.insert(" ")
        clock_marker = pynini.closure(pynutil.delete(" ", weight=0.0001), 0, 1) + pynutil.delete("uhr")
        hour_t = self.tag_field("hours", self.hour_graph) + clock_marker
        hm_t = hour_t + separator + self.tag_field("minutes", self.number_graph)

        def counted(singular, plural):
            graph = (
                pynini.cross("1", "eine " + singular)
                | (values - "1") @ numbers + pynutil.insert(" " + plural)
            )
            return inverse(graph)

        minute_count = counted("minute", "minuten")
        second_count = counted("sekunde", "sekunden")
        hms_t = (
            hour_t + separator + self.tag_field("minutes", minute_count)
            + separator + self.tag_field("seconds", second_count)
        )
        past_minutes = inverse(tn_time.past_minute_graph)
        to_minutes = inverse(tn_time.to_minute_graph)
        night = pynutil.add_weight(
            pynini.string_file(get_abs_path("german/data/time/hour_to_night.tsv")).invert(), 0.0001,
        )
        night_hours = values @ self.build_rule(night, l="[BOS]", r="[EOS]")
        next_hours = night_hours @ pynini.string_file(get_abs_path("german/data/time/hour_to.tsv")).invert()
        past_hours = inverse(night_hours @ numbers)
        to_hours = inverse(next_hours @ numbers)
        past_t = (
            self.tag_field("minutes", past_minutes) + separator + pynutil.delete("nach")
            + separator + self.tag_field("hours", past_hours)
        )
        to_t = (
            self.tag_field("minutes", to_minutes) + separator + pynutil.delete("vor")
            + separator + self.tag_field("hours", to_hours)
        )
        half_t = pynutil.delete("halb") + separator + self.tag_field("hours", to_hours)
        zone_t = pynini.closure(separator + self.tag_field("zone", self.zone_graph), 0, 1)
        self.tagger = self.add_tokens((
            pynutil.insert('kind: "hour" ') + hour_t
            | pynutil.insert('kind: "clock" ') + (hm_t | hms_t)
            | pynutil.add_weight(pynutil.insert('kind: "past" ') + past_t, 0.0001)
            | pynutil.add_weight(pynutil.insert('kind: "to" ') + to_t, 0.0001)
            | pynutil.add_weight(pynutil.insert('kind: "half" ') + half_t, 0.0001)
        ) + zone_t)

        pad = self.DIGIT + self.DIGIT | pynutil.insert("0") + self.DIGIT
        hour_v = self.verbalize_field("hours", self.hour_graph)
        hm_v = self.verbalize_field("hours", self.hour_graph @ pad) + pynutil.insert(":")
        hm_v += self.verbalize_field("minutes", self.number_graph @ pad, leading_space=True)
        hms_v = self.verbalize_field("hours", self.hour_graph @ pad) + pynutil.insert(":")
        hms_v += self.verbalize_field("minutes", minute_count @ pad, leading_space=True)
        hms_v += pynutil.insert(":") + self.verbalize_field("seconds", second_count @ pad, leading_space=True)
        past_v = self.verbalize_field("hours", past_hours @ pad) + pynutil.insert(":")
        past_v += self.verbalize_field("minutes", past_minutes @ pad, leading_space=True)
        to_v = self.verbalize_field("hours", to_hours @ pad) + pynutil.insert(":")
        to_v += self.verbalize_field("minutes", to_minutes @ pad, leading_space=True)
        half_v = self.verbalize_field("hours", to_hours @ pad) + pynutil.insert(":30")
        zone_v = pynini.closure(
            pynutil.insert(" ") + self.verbalize_field("zone", self.zone_graph, leading_space=True), 0, 1,
        )
        self.verbalizer = self.delete_tokens((
            pynutil.delete('kind: "hour" ') + hour_v
            | pynutil.delete('kind: "clock" ') + (hm_v | hms_v)
            | pynutil.delete('kind: "past" ') + past_v
            | pynutil.delete('kind: "to" ') + to_v
            | pynutil.delete('kind: "half" ') + half_v
        ) + pynutil.insert(" Uhr") + zone_v)
