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

from pynini import cdrewrite, cross, string_file, string_map, union
from pynini.lib.pynutil import delete, insert

from tn.english.rules.cardinal import Cardinal
from tn.processor import Processor
from tn.utils import get_abs_path, load_labels, augment_labels_with_punct_at_end


class Time(Processor):

    def __init__(self, deterministic: bool = False):
        """
        Args:
            deterministic: if True will provide a single transduction option,
                for False multiple transduction are generated (used for audio-based normalization)
        """
        super().__init__("time", ordertype="en_tn")
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
        suffix_labels = load_labels(get_abs_path("english/data/time/suffix.tsv"))
        suffix_labels.extend(augment_labels_with_punct_at_end(suffix_labels))
        suffix_graph = string_map(suffix_labels)

        time_zone_graph = string_file(get_abs_path("english/data/time/zone.tsv"))

        # only used for < 1000 thousand -> 0 weight
        cardinal = Cardinal(self.deterministic).graph

        labels_hour = [str(x) for x in range(0, 24)]
        labels_minute_single = [str(x) for x in range(1, 10)]
        labels_minute_double = [str(x) for x in range(10, 60)]

        delete_leading_zero_to_double_digit = (self.DIGIT + self.DIGIT) | (
            delete("0").ques + self.DIGIT
        )

        graph_hour = (
            delete_leading_zero_to_double_digit @ union(*labels_hour) @ cardinal
        )

        graph_minute_single = union(*labels_minute_single) @ cardinal
        graph_minute_double = union(*labels_minute_double) @ cardinal

        final_graph_hour = insert('hours: "') + graph_hour + insert('"')
        final_graph_minute = (
            insert('minutes: "')
            + (
                cross("0", "o") + insert(" ") + graph_minute_single
                | graph_minute_double
            )
            + insert('"')
        )
        final_graph_second = (
            insert('seconds: "')
            + (
                cross("0", "o") + insert(" ") + graph_minute_single
                | graph_minute_double
            )
            + insert('"')
        )
        final_suffix = insert('suffix: "') + suffix_graph + insert('"')
        final_suffix_optional = (self.DELETE_SPACE + insert(" ") + final_suffix).ques
        final_time_zone_optional = (
            self.DELETE_SPACE
            + insert(" ")
            + insert('zone: "')
            + time_zone_graph
            + insert('"')
        ).ques

        # 2:30 pm, 02:30, 2:00
        graph_hm = (
            final_graph_hour
            + delete(":")
            + (delete("00") | insert(" ") + final_graph_minute)
            + final_suffix_optional
            + final_time_zone_optional
        )

        # 10:30:05 pm,
        graph_hms = (
            final_graph_hour
            + delete(":")
            + (cross("00", ' minutes: "zero"') | insert(" ") + final_graph_minute)
            + delete(":")
            + (cross("00", ' seconds: "zero"') | insert(" ") + final_graph_second)
            + final_suffix_optional
            + final_time_zone_optional
        )

        # 2.xx pm/am
        graph_hm2 = (
            final_graph_hour
            + delete(".")
            + (delete("00") | insert(" ") + final_graph_minute)
            + self.DELETE_SPACE
            + insert(" ")
            + final_suffix
            + final_time_zone_optional
        )
        # 2 pm est
        graph_h = (
            final_graph_hour
            + self.DELETE_SPACE
            + insert(" ")
            + final_suffix
            + final_time_zone_optional
        )
        final_graph = (graph_hm | graph_h | graph_hm2 | graph_hms).optimize()

        final_graph = self.add_tokens(final_graph)
        self.tagger = final_graph.optimize()

    def build_verbalizer(self):
        """
        Finite state transducer for verbalizing time, e.g.
            time { hours: "twelve" minutes: "thirty" suffix: "a m" zone: "e s t" } -> twelve thirty a m e s t
            time { hours: "twelve" } -> twelve o'clock
        """
        hour = (
            delete("hours:")
            + self.DELETE_SPACE
            + delete('"')
            + self.NOT_QUOTE.plus
            + delete('"')
        )
        minute = (
            delete("minutes:")
            + self.DELETE_SPACE
            + delete('"')
            + self.NOT_QUOTE.plus
            + delete('"')
        )
        suffix = (
            delete("suffix:")
            + self.DELETE_SPACE
            + delete('"')
            + self.NOT_QUOTE.plus
            + delete('"')
        )
        optional_suffix = (self.DELETE_SPACE + insert(" ") + suffix).ques
        zone = (
            delete("zone:")
            + self.DELETE_SPACE
            + delete('"')
            + self.NOT_QUOTE.plus
            + delete('"')
        )
        optional_zone = (self.DELETE_SPACE + insert(" ") + zone).ques
        second = (
            delete("seconds:")
            + self.DELETE_SPACE
            + delete('"')
            + self.NOT_QUOTE.plus
            + delete('"')
        )
        graph_hms = (
            hour
            + insert(" hours ")
            + self.DELETE_SPACE
            + minute
            + insert(" minutes and ")
            + self.DELETE_SPACE
            + second
            + insert(" seconds")
            + optional_suffix
            + optional_zone
        )
        graph_hms @= cdrewrite(
            delete("o ")
            | cross("one minutes", "one minute")
            | cross("one seconds", "one second")
            | cross("one hours", "one hour"),
            union(" ", "[BOS]"),
            "",
            self.VSIGMA,
        )
        graph = (
            hour
            + self.DELETE_SPACE
            + insert(" ")
            + minute
            + optional_suffix
            + optional_zone
        )
        graph |= hour + insert(" ") + insert("o'clock") + optional_zone
        graph |= hour + self.DELETE_SPACE + insert(" ") + suffix + optional_zone
        graph |= graph_hms
        delete_tokens = self.delete_tokens(graph)
        self.verbalizer = delete_tokens.optimize()
