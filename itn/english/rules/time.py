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

from pynini import closure, cross, invert, string_file, union
from pynini.lib.pynutil import delete, insert

TO_OR_TILL = union("to", "till")

from itn.english.rules.cardinal import Cardinal
from tn.processor import Processor
from itn.utils import get_abs_path


def _num_to_word(n):
    ones = [
        "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen",
        "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"
    ]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    if n < 20:
        return ones[n]
    return tens[n // 10] + (" " + ones[n % 10] if n % 10 else "")


class Time(Processor):

    def __init__(self, cardinal=None):
        super().__init__(name="time", ordertype="itn")
        self.cardinal = cardinal or Cardinal()
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        time_suffix = string_file(get_abs_path("english/data/time/time_suffix.tsv"))
        time_zone = invert(string_file(get_abs_path("english/data/time/time_zone.tsv")))
        to_hour = string_file(get_abs_path("english/data/time/to_hour.tsv"))
        minute_to = string_file(get_abs_path("english/data/time/minute_to.tsv"))
        ds = delete(" ")

        hour_all = union(*[cross(_num_to_word(x), f"{x:02d}") for x in range(0, 24) if _num_to_word(x)])
        hour_12 = union(*[cross(_num_to_word(x), f"{x:02d}") for x in range(1, 13)])
        graph_min_single = union(*[cross(_num_to_word(x), f"0{x}") for x in range(1, 10)])
        graph_min_double = union(*[cross(_num_to_word(x), str(x)) for x in range(10, 60)])
        graph_min_verbose = cross("half", "30") | cross("quarter", "15")

        # minutes without zero-padding (for minute_to composition)
        min_single_raw = union(*[cross(_num_to_word(x), str(x)) for x in range(1, 10)])
        min_double_raw = graph_min_double  # already no padding

        oclock = cross("o'clock", "") | cross("o' clock", "") | cross("o clock", "") | cross("oclock", "") | cross(
            "hundred hours", "")

        self.hour = hour_all | hour_12
        self.hour_to = delete(TO_OR_TILL) + ds + to_hour
        self.minute_oclock = oclock + insert("00")
        self.minute = delete("o") + ds + graph_min_single | graph_min_double
        self.minute_past = ((graph_min_single | graph_min_double | graph_min_verbose) + ds + delete("past"))
        self.minute_quarter_to = cross("quarter", "45")
        self.minute_to = (((min_single_raw | min_double_raw) @ minute_to) +
                          closure(ds + delete("min") + delete("ute").ques + delete("s").ques, 0, 1))
        self.noon = time_suffix
        self.zone = time_zone

        hour = self.tag_field("hour", self.hour)
        hour12 = self.tag_field("hour", hour_12)
        suffix = ds + insert(" ") + self.tag_field("noon", self.noon)
        zone = ds + insert(" ") + self.tag_field("zone", self.zone)
        zone_opt = closure(zone, 0, 1)

        # "eight oclock" / "eight oclock gmt"
        graph_oclock = (hour + ds + insert(" ") + self.tag_field("minute", self.minute_oclock) + zone_opt)
        # "two o five"
        graph_o_min = (hour + ds + insert(" ") + self.tag_field("minute", delete("o") + ds + graph_min_single))
        # "two pm" / "three am est"
        graph_h_suffix = hour + insert(' minute: ""') + suffix + zone_opt
        # "two thirty am"
        graph_hm_suffix = (hour + ds + insert(" ") + self.tag_field("minute", graph_min_double) + suffix + zone_opt)
        # "two thirty" (1-12 only, no suffix)
        graph_hm = hour12 + ds + insert(" ") + self.tag_field("minute", graph_min_double)
        # "eleven o six pm"
        graph_o_min_suffix = (hour + ds + insert(" ") + self.tag_field("minute",
                                                                       delete("o") + ds + graph_min_single) + suffix +
                              zone_opt)
        # "half past two", "quarter past two", "ten past four"
        graph_past = (self.tag_field("minute", self.minute_past) + ds + insert(" ") + hour)
        # "quarter to one" / "quarter till one" => 12:45
        graph_quarter_to = (self.tag_field("minute", self.minute_quarter_to) + ds + insert(" ") +
                            self.tag_field("hour", self.hour_to) + insert(' style: "to"'))
        # "ten to eleven pm" / "ten till eleven pm" => 10:50 p.m.
        graph_min_to = (self.tag_field("minute", self.minute_to) + ds + insert(" ") + self.tag_field("hour", self.hour_to) +
                        suffix + insert(' style: "to"'))

        final_graph = (graph_oclock | graph_o_min | graph_h_suffix
                       | graph_hm_suffix | graph_hm | graph_o_min_suffix
                       | graph_past | graph_quarter_to | graph_min_to)
        self.tagger = self.add_tokens(final_graph)

    def build_verbalizer(self):
        hour = self.verbalize_field("hour", self.hour | self.hour_to)
        minute = self.verbalize_field(
            "minute",
            self.minute
            | self.minute_oclock
            | self.minute_past
            | self.minute_quarter_to
            | self.minute_to
            | insert("00"),
            leading_space=True,
        )
        noon = self.verbalize_field("noon", self.noon, leading_space=True)
        zone = self.verbalize_field("zone", self.zone, leading_space=True)
        graph = hour + insert(":") + self.DELETE_SPACE + minute
        graph += closure(insert(" ") + self.DELETE_SPACE + noon, 0, 1)
        graph += closure(insert(" ") + self.DELETE_SPACE + zone, 0, 1)
        style_to = delete(' style: "to"')
        graph_regular = graph
        minute_to = self.verbalize_field(
            "minute",
            self.minute_quarter_to | self.minute_to,
            leading_space=True,
        )
        hour_to = self.verbalize_field("hour", self.hour_to)
        graph_to = hour_to + insert(":") + self.DELETE_SPACE + minute_to
        graph_to += closure(insert(" ") + self.DELETE_SPACE + noon, 0, 1)
        graph_to += closure(insert(" ") + self.DELETE_SPACE + zone, 0, 1)
        graph_to += style_to
        self.verbalizer = self.delete_tokens(graph_regular | graph_to)
