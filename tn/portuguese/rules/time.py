# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pynini.lib.pynutil import delete, insert

from tn.processor import Processor
from tn.utils import get_abs_path, load_labels


TOKEN_ORDER = {}


def resolve_period_hour(hour, tail, allowed):
    """Returns the clock hour spoken before a written day period (NeMo policy)."""
    if tail == "manhã":
        return hour
    if tail == "tarde":
        if hour in allowed:
            return 12 if hour == 12 else hour - 12
        return hour
    if tail == "noite":
        return hour - 12 if hour in allowed else hour
    return hour


class Time(Processor):
    """Raw clock fields, e.g. 14h30 da tarde -> hours "14" minutes "30" suffix "da tarde".

    The verbalizer converts 24-hour clock values to the 12-hour reading only
    when the written day period matches, e.g. 15h da tarde -> três horas.
    """

    def __init__(self, cardinal):
        super().__init__("time", token_orders=TOKEN_ORDER)

        def spell(graph, value):
            return pynini.shortestpath(pynini.accep(str(value)) @ graph).string()

        def hour_phrase(value):
            return "uma hora" if value == 1 else spell(cardinal.graph_fem, value) + " horas"

        written_hours = [(str(h), h) for h in range(24)] + [("0" + str(h), h) for h in range(10)]
        periods = []
        for row in load_labels(get_abs_path("portuguese/data/time/day_period_suffix.tsv")):
            tail, spoken, first, last = row[0], row[1], int(row[2]), int(row[3])
            allowed = set(range(first, last + 1))
            hours = pynini.string_map([
                (written, hour_phrase(resolve_period_hour(h, tail, allowed))) for written, h in written_hours
            ])
            prep = spoken.split()[0]
            raw = pynini.accep(prep) + self.SPACE.plus + tail
            periods.append((hours.optimize(), raw, pynini.cross(raw, spoken)))
        hours_24 = pynini.string_map([(written, hour_phrase(h)) for written, h in written_hours]).optimize()

        minutes = pynini.string_map([("{:02d}".format(m), spell(cardinal.graph, m)) for m in range(1, 60)])
        zero = pynini.cross("00", cardinal.zero)
        clock = pynini.union(":", ".")
        letter = pynini.union("h", "H")

        hour_t = self.tag_field("hours", hours_24)
        minute_t = insert(" ") + self.tag_field("minutes", minutes | zero)
        second_t = insert(" ") + self.tag_field("seconds", minutes | zero)
        clock_t = (
            hour_t + delete(clock) + minute_t
            | hour_t + delete(letter) + minute_t.ques
            | hour_t + delete(clock) + minute_t + delete(clock) + second_t
        )
        suffix_raw = pynini.union(*[raw for _, raw, _ in periods])
        suffix_t = self.DELETE_SPACE + insert(" ") + self.tag_field("suffix", suffix_raw)
        self.tagger = self.add_tokens(clock_t + suffix_t.ques)

        minute_v = (
            self.verbalize_field("minutes", delete("00"), leading_space=True)
            | insert(" e ") + self.verbalize_field("minutes", minutes, leading_space=True)
        )
        seconds_v = (
            insert(" e ") + self.verbalize_field("minutes", minutes | zero, leading_space=True)
            + insert(" minutos e ") + self.verbalize_field("seconds", minutes | zero, leading_space=True)
            + insert(" segundos")
        )
        rest_v = (minute_v.ques | seconds_v).optimize()
        graphs = [self.verbalize_field("hours", hours_24) + rest_v]
        for hours, _, suffix in periods:
            graphs.append(
                self.verbalize_field("hours", hours) + rest_v
                + insert(" ") + self.verbalize_field("suffix", suffix, leading_space=True)
            )
        self.verbalizer = self.delete_tokens(pynini.union(*graphs))
