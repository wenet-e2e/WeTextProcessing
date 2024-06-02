# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import pynini
from pynini.lib import pynutil

from tn.processor import Processor
from tn.utils import get_abs_path, load_labels, augment_labels_with_punct_at_end
from tn.english.rules.cardinal import Cardinal


class Time(Processor):

    def __init__(self, deterministic: bool = False):
        """
        Args:
            deterministic: if True will provide a single transduction option,
                for False multiple transduction are generated (used for audio-based normalization)
        """
        super().__init__('time', ordertype="en_tn")
        self.deterministic = deterministic
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        """
        Finite state transducer for classifying time, e.g.
            12:30 a.m. est -> time { hours: "twelve" minutes: "thirty" suffix: "a m" zone: "e s t" }
            2.30 a.m. -> time { hours: "two" minutes: "thirty" suffix: "a m" }
            02.30 a.m. -> time { hours: "two" minutes: "thirty" suffix: "a m" }
            2.00 a.m. -> time { hours: "two" suffix: "a m" }
            2 a.m. -> time { hours: "two" suffix: "a m" }
            02:00 -> time { hours: "two" }
            2:00 -> time { hours: "two" }
            10:00:05 a.m. -> time { hours: "ten" minutes: "zero" seconds: "five" suffix: "a m" }
        """
        suffix_labels = load_labels(
            get_abs_path("english/data/time/suffix.tsv"))
        suffix_labels.extend(augment_labels_with_punct_at_end(suffix_labels))
        suffix_graph = pynini.string_map(suffix_labels)

        time_zone_graph = pynini.string_file(
            get_abs_path("english/data/time/zone.tsv"))

        # only used for < 1000 thousand -> 0 weight
        cardinal = Cardinal(self.deterministic).graph

        labels_hour = [str(x) for x in range(0, 24)]
        labels_minute_single = [str(x) for x in range(1, 10)]
        labels_minute_double = [str(x) for x in range(10, 60)]

        delete_leading_zero_to_double_digit = (self.DIGIT + self.DIGIT) | (
            pynini.closure(pynutil.delete("0"), 0, 1) + self.DIGIT)

        graph_hour = delete_leading_zero_to_double_digit @ pynini.union(
            *labels_hour) @ cardinal

        graph_minute_single = pynini.union(*labels_minute_single) @ cardinal
        graph_minute_double = pynini.union(*labels_minute_double) @ cardinal

        final_graph_hour = pynutil.insert(
            "hours: \"") + graph_hour + pynutil.insert("\"")
        final_graph_minute = (
            pynutil.insert("minutes: \"") +
            (pynini.cross("0", "o") + self.INSERT_SPACE + graph_minute_single
             | graph_minute_double) + pynutil.insert("\""))
        final_graph_second = (
            pynutil.insert("seconds: \"") +
            (pynini.cross("0", "o") + self.INSERT_SPACE + graph_minute_single
             | graph_minute_double) + pynutil.insert("\""))
        final_suffix = pynutil.insert(
            "suffix: \"") + suffix_graph + pynutil.insert("\"")
        final_suffix_optional = pynini.closure(
            self.DELETE_SPACE + self.INSERT_SPACE + final_suffix, 0, 1)
        final_time_zone_optional = pynini.closure(
            self.DELETE_SPACE + self.INSERT_SPACE +
            pynutil.insert("zone: \"") + time_zone_graph +
            pynutil.insert("\""),
            0,
            1,
        )

        # 2:30 pm, 02:30, 2:00
        graph_hm = (
            final_graph_hour + pynutil.delete(":") +
            (pynutil.delete("00") | self.INSERT_SPACE + final_graph_minute) +
            final_suffix_optional + final_time_zone_optional)

        # 10:30:05 pm,
        graph_hms = (final_graph_hour + pynutil.delete(":") +
                     (pynini.cross("00", " minutes: \"zero\"")
                      | self.INSERT_SPACE + final_graph_minute) +
                     pynutil.delete(":") +
                     (pynini.cross("00", " seconds: \"zero\"")
                      | self.INSERT_SPACE + final_graph_second) +
                     final_suffix_optional + final_time_zone_optional)

        # 2.xx pm/am
        graph_hm2 = (
            final_graph_hour + pynutil.delete(".") +
            (pynutil.delete("00") | self.INSERT_SPACE + final_graph_minute) +
            self.DELETE_SPACE + self.INSERT_SPACE + final_suffix +
            final_time_zone_optional)
        # 2 pm est
        graph_h = final_graph_hour + self.DELETE_SPACE + self.INSERT_SPACE + final_suffix + final_time_zone_optional
        final_graph = (graph_hm | graph_h | graph_hm2 | graph_hms).optimize()

        final_graph = self.add_tokens(final_graph)
        self.tagger = final_graph.optimize()

    def build_verbalizer(self):
        """
        Finite state transducer for verbalizing time, e.g.
            time { hours: "twelve" minutes: "thirty" suffix: "a m" zone: "e s t" } -> twelve thirty a m e s t
            time { hours: "twelve" } -> twelve o'clock
        """
        hour = (pynutil.delete("hours:") + self.DELETE_SPACE +
                pynutil.delete("\"") + pynini.closure(self.NOT_QUOTE, 1) +
                pynutil.delete("\""))
        minute = (pynutil.delete("minutes:") + self.DELETE_SPACE +
                  pynutil.delete("\"") + pynini.closure(self.NOT_QUOTE, 1) +
                  pynutil.delete("\""))
        suffix = (pynutil.delete("suffix:") + self.DELETE_SPACE +
                  pynutil.delete("\"") + pynini.closure(self.NOT_QUOTE, 1) +
                  pynutil.delete("\""))
        optional_suffix = pynini.closure(
            self.DELETE_SPACE + self.INSERT_SPACE + suffix, 0, 1)
        zone = (pynutil.delete("zone:") + self.DELETE_SPACE +
                pynutil.delete("\"") + pynini.closure(self.NOT_QUOTE, 1) +
                pynutil.delete("\""))
        optional_zone = pynini.closure(
            self.DELETE_SPACE + self.INSERT_SPACE + zone, 0, 1)
        second = (pynutil.delete("seconds:") + self.DELETE_SPACE +
                  pynutil.delete("\"") + pynini.closure(self.NOT_QUOTE, 1) +
                  pynutil.delete("\""))
        graph_hms = (hour + pynutil.insert(" hours ") + self.DELETE_SPACE +
                     minute + pynutil.insert(" minutes and ") +
                     self.DELETE_SPACE + second + pynutil.insert(" seconds") +
                     optional_suffix + optional_zone)
        graph_hms @= pynini.cdrewrite(
            pynutil.delete("o ")
            | pynini.cross("one minutes", "one minute")
            | pynini.cross("one seconds", "one second")
            | pynini.cross("one hours", "one hour"),
            pynini.union(" ", "[BOS]"),
            "",
            pynini.closure(self.VCHAR),
        )
        graph = hour + self.DELETE_SPACE + self.INSERT_SPACE + minute + optional_suffix + optional_zone
        graph |= hour + self.INSERT_SPACE + pynutil.insert(
            "o'clock") + optional_zone
        graph |= hour + self.DELETE_SPACE + self.INSERT_SPACE + suffix + optional_zone
        graph |= graph_hms
        delete_tokens = self.delete_tokens(graph)
        self.verbalizer = delete_tokens.optimize()
