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
from pynini.lib.pynutil import add_weight, delete, insert

from itn.english.rules.cardinal import Cardinal
from tn.processor import Processor
from tn.utils import get_abs_path


def _num_to_word(n):
    ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
            "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
            "seventeen", "eighteen", "nineteen"]
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
        cardinal_graph = add_weight(self.cardinal.graph_no_exception, -0.7)
        time_suffix = string_file(get_abs_path("../itn/english/data/time/time_suffix.tsv"))
        time_zone = invert(string_file(get_abs_path("../itn/english/data/time/time_zone.tsv")))
        to_hour = string_file(get_abs_path("../itn/english/data/time/to_hour.tsv"))
        minute_to = string_file(get_abs_path("../itn/english/data/time/minute_to.tsv"))
        ds = delete(" ")

        hour_all = union(*[cross(_num_to_word(x), f"{x:02d}") for x in range(0, 24) if _num_to_word(x)])
        hour_12 = union(*[cross(_num_to_word(x), f"{x:02d}") for x in range(1, 13)])
        graph_min_single = union(*[cross(_num_to_word(x), f"0{x}") for x in range(1, 10)])
        graph_min_double = union(*[cross(_num_to_word(x), str(x)) for x in range(10, 60)])
        graph_min_verbose = cross("half", "30") | cross("quarter", "15")

        # minutes without zero-padding (for minute_to composition)
        min_single_raw = union(*[cross(_num_to_word(x), str(x)) for x in range(1, 10)])
        min_double_raw = graph_min_double  # already no padding

        oclock = cross("o'clock", "") | cross("oclock", "") | cross("hundred hours", "")

        hour = insert('hour: "') + hour_all + insert('"')
        hour12 = insert('hour: "') + hour_12 + insert('"')
        suffix = ds + insert(' noon: "') + time_suffix + insert('"')
        zone = ds + insert(' zone: "') + time_zone + insert('"')
        zone_opt = closure(zone, 0, 1)

        # "eight oclock" / "eight oclock gmt"
        graph_oclock = hour + ds + insert(' minute: "') + oclock + insert('00"') + zone_opt
        # "two o five"
        graph_o_min = hour + ds + insert(' minute: "') + delete("o") + ds + graph_min_single + insert('"')
        # "two pm" / "three am est"
        graph_h_suffix = hour + insert(' minute: "00"') + suffix + zone_opt
        # "two thirty am"
        graph_hm_suffix = (
            hour + ds + insert(' minute: "') + graph_min_double + insert('"') + suffix + zone_opt
        )
        # "two thirty" (1-12 only, no suffix)
        graph_hm = hour12 + ds + insert(' minute: "') + graph_min_double + insert('"')
        # "eleven o six pm"
        graph_o_min_suffix = (
            hour + ds + insert(' minute: "') + delete("o") + ds + graph_min_single + insert('"') + suffix + zone_opt
        )
        # "half past two", "quarter past two"
        graph_past = (
            insert('minute: "') + graph_min_verbose + insert('"') + ds + delete("past") + ds + hour
        )
        # "quarter to one" => 12:45
        graph_quarter_to = (
            insert('minute: "') + cross("quarter", "45") + insert('"')
            + ds + delete("to") + ds
            + insert('hour: "') + to_hour + insert('"')
        )
        # "ten to eleven pm" => 10:50 p.m.
        graph_min_to = (
            insert('minute: "')
            + ((min_single_raw | min_double_raw) @ minute_to)
            + insert('"')
            + closure(ds + delete("min") + delete("ute").ques + delete("s").ques, 0, 1)
            + ds + delete("to") + ds
            + insert('hour: "') + to_hour + insert('"')
            + suffix
        )

        final_graph = (
            graph_oclock | graph_o_min | graph_h_suffix
            | graph_hm_suffix | graph_hm | graph_o_min_suffix
            | graph_past | graph_quarter_to | graph_min_to
        )
        self.tagger = self.add_tokens(final_graph)

    def build_verbalizer(self):
        hour = delete('hour: "') + self.NOT_QUOTE.plus + delete('"')
        minute = delete(' minute: "') + self.NOT_QUOTE.plus + delete('"')
        noon = delete(' noon: "') + self.NOT_QUOTE.plus + delete('"')
        zone = delete(' zone: "') + self.NOT_QUOTE.plus + delete('"')
        graph = hour + insert(":") + self.DELETE_SPACE + minute
        graph += closure(insert(" ") + self.DELETE_SPACE + noon, 0, 1)
        graph += closure(insert(" ") + self.DELETE_SPACE + zone, 0, 1)
        self.verbalizer = self.delete_tokens(graph)
