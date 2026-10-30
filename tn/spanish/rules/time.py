# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from tn.spanish.rules.utils import (
    DELETE_SPACE,
    DIGIT,
    ONES,
    data_file,
    data_labels,
    rewrite,
)

TOKEN_ORDER = {"time": ["hours", "minutes", "seconds", "suffix", "zone"]}


def _labels(first, last):
    return pynini.union(*[str(number) for number in range(first, last + 1)])


class Time(Processor):
    """Raw clock fields, day-period suffix and zone; V reads them in Spanish."""

    def __init__(self, cardinal, deterministic=True):
        super().__init__("time", token_orders=TOKEN_ORDER)
        self.deterministic = deterministic
        suffix = data_file("time/time_suffix.tsv")
        zones = data_file("time/time_zone.tsv")
        leading_zero = pynini.closure(pynutil.delete("0") | (DIGIT - "0"), 0, 1) + DIGIT
        hour_24 = (pynini.closure(DIGIT, 1, 2) @ leading_zero @ _labels(0, 24)).optimize()
        hour_12 = (pynini.closure(DIGIT, 1, 2) @ leading_zero @ _labels(1, 12)).optimize()
        minute = (pynini.closure(DIGIT, 1, 2) @ leading_zero @ _labels(1, 59)).optimize()

        def feminine(graph):
            return rewrite(graph @ cardinal.graph, pynini.cross(ONES, "una"))

        hours_24 = feminine(hour_24)
        hours_12 = feminine(hour_12)
        minutes = (minute @ cardinal.graph).optimize()

        # Zone offsets: "gmt-4:30" is "g m t menos cuatro treinta".
        offset = (DIGIT @ rewrite(cardinal.graph, pynini.cross(ONES, "uno"))) | (
            hours_24 + pynini.cross(pynini.union(".", ":"), " ") + minutes
        )
        optional_space = pynutil.delete(pynini.closure(" ", 0, 1)) + pynutil.insert(" ")
        signs = pynini.string_map([("+", "más"), ("-", "menos")])
        zone_graph = (zones + pynini.closure(optional_space + signs + optional_space + offset, 0, 1)).optimize()

        # Tagger: delimiters and the h/min/seg markers are structure only.
        space = pynutil.insert(" ")
        delimiter = pynutil.delete(pynini.union(".", ":")) + space
        hours_t = self.tag_field("hours", hour_24)
        hours_12_t = self.tag_field("hours", hour_12)
        minutes_t = self.tag_field("minutes", minute | "00")
        seconds_t = self.tag_field("seconds", minute | "00")
        hour_mark = DELETE_SPACE + pynutil.delete("h")
        suffix_t = DELETE_SPACE + space + self.tag_field("suffix", suffix)
        zone_t = pynini.closure(DELETE_SPACE + space + self.tag_field("zone", zone_graph), 0, 1)
        clock_t = minutes_t + pynini.closure(delimiter + seconds_t, 0, 1)
        graph = hours_t + delimiter + clock_t + pynini.closure(hour_mark, 0, 1) + zone_t
        graph |= (
            hours_t + hour_mark + DELETE_SPACE + space + minutes_t + DELETE_SPACE + pynutil.delete(pynini.union("min", "m"))
            + pynini.closure(
                DELETE_SPACE + space + seconds_t + DELETE_SPACE + pynutil.delete(pynini.union("seg", "s")), 0, 1,
            )
            + zone_t
        )
        graph |= hours_12_t + delimiter + clock_t + suffix_t + zone_t
        graph |= hours_t + hour_mark + zone_t | hours_12_t + suffix_t + zone_t
        self.tagger = self.add_tokens(graph)

        # Verbalizer.
        alt_minutes = data_file("time/alt_minutes.tsv")
        changed = rewrite(minutes, alt_minutes, "[BOS]", "[EOS]")
        spoken_minutes = changed if deterministic else (changed | pynutil.add_weight(minutes, 0.001))
        zone_v = pynini.closure(space + self.verbalize_field("zone", zone_graph, leading_space=True), 0, 1)
        no_minutes = self.verbalize_field("minutes", pynutil.delete("00"), leading_space=True)
        no_seconds = pynini.closure(self.verbalize_field("seconds", pynutil.delete("00"), leading_space=True), 0, 1)
        minutes_v = (
            pynutil.insert(" y ") + self.verbalize_field("minutes", spoken_minutes, leading_space=True) | no_minutes
        )
        clock_v = pynini.closure(minutes_v + no_seconds, 0, 1)

        def times(name):
            return [row[0] for row in data_labels("time/" + name + ".tsv")]

        mid_phrases = [("pm", "del mediodía"), ("am", "de la noche")]
        if not deterministic:
            mid_phrases += [("pm", "de la mañana"), ("pm", "del día"), ("am", "de la medianoche")]
        periods = [
            (times("morning_times"), pynini.cross("am", "de la mañana")),
            (times("afternoon_times"), pynini.cross("pm", "de la tarde")),
            (times("evening_times"), pynini.cross("pm", "de la noche")),
            (["doce"], pynini.string_map(mid_phrases)),
        ]

        def suffix_v(phrase):
            return space + self.verbalize_field("suffix", suffix @ phrase, leading_space=True)

        def with_periods(hours, middle):
            """Day-period phrases depend on the spoken hour."""
            return pynini.union(*[
                self.verbalize_field("hours", (hours @ pynini.union(*words)).optimize()) + middle + suffix_v(phrase)
                for words, phrase in periods
            ])

        graph = self.verbalize_field("hours", hours_24) + clock_v + zone_v
        graph |= with_periods(hours_12, clock_v) + zone_v
        # "tres horas dos minutos y treinta y dos segundos".
        seconds_v = space + self.verbalize_field("seconds", minutes, leading_space=True) + pynutil.insert(" segundos")
        graph |= (
            self.verbalize_field("hours", hours_24) + pynutil.insert(" horas")
            + (
                space + self.verbalize_field("minutes", minutes, leading_space=True) + pynutil.insert(" minutos y")
                | no_minutes + pynutil.insert(" y")
            )
            + seconds_v + zone_v
        )

        if not deterministic:
            # "tres menos cuarto" and "cuarto para las tres".
            def shifted(name, hours):
                return (hours @ pynini.invert(data_file("time/" + name + ".tsv"))).optimize()

            shift_24 = shifted("hour_to_24", hour_24)
            shift_12 = shifted("hour_to_12", hour_12)
            minutes_to = (minute @ data_file("time/minute_to.tsv") @ cardinal.graph).optimize()
            minutes_to = rewrite(minutes_to, alt_minutes, "[BOS]", "[EOS]") | minutes_to
            minutes_to_v = self.verbalize_field("minutes", minutes_to, leading_space=True)
            menos = pynutil.insert(" menos ") + minutes_to_v
            graph |= pynutil.add_weight(self.verbalize_field("hours", feminine(shift_24)) + menos + zone_v, 0.01)
            graph |= pynutil.add_weight(with_periods(feminine(shift_12), menos) + zone_v, 0.01)

            def spoken(fst):
                return pynini.shortestpath(fst).string()

            # The hour is spoken after the minutes, so each written hour is a branch.
            para = []
            for hours, name, last, first in ((hour_24, "hour_to_24", 24, 0), (hour_12, "hour_to_12", 12, 1)):
                shift = pynini.invert(data_file("time/" + name + ".tsv"))
                for value in range(first, last + 1):
                    word = spoken(feminine(pynini.accep(str(value)) @ shift))
                    article = " para la " if word == "una" else " para las "
                    written = pynini.project(hours @ pynini.accep(str(value)), "input").optimize()
                    branch = (
                        self.verbalize_field("hours", pynutil.delete(written)) + minutes_to_v
                        + pynutil.insert(article + word)
                    )
                    if last == 24:
                        para.append(branch + zone_v)
                        continue
                    for words, phrase in periods:
                        if word in words:
                            para.append(branch + suffix_v(phrase) + zone_v)
            graph |= pynutil.add_weight(pynini.union(*para), 0.01)
        self.verbalizer = self.delete_tokens(graph)
