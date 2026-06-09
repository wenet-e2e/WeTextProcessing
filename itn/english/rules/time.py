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

from pynini import closure, cross, string_file, union
from pynini.lib.pynutil import delete, insert

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
        cardinal_graph = self.cardinal.graph
        time_suffix = string_file(get_abs_path("../itn/english/data/time/time_suffix.tsv"))
        time_zone = string_file(get_abs_path("../itn/english/data/time/time_zone.tsv"))
        ds = delete(" ")

        # hours: 0-23, only valid hour words, with zero-padding
        hour_labels = [_num_to_word(x) for x in range(0, 24) if _num_to_word(x)]
        hour_padded = union(*[cross(_num_to_word(x), f"{x:02d}") for x in range(0, 24) if _num_to_word(x)])
        # minutes: 1-9 (single), 10-59 (double)
        min_single = [_num_to_word(x) for x in range(1, 10)]
        min_double = [_num_to_word(x) for x in range(10, 60)]
        graph_min_single = union(*[cross(_num_to_word(x), f"{x:02d}") for x in range(1, 10)])
        graph_min_double = union(*[cross(_num_to_word(x), str(x)) for x in range(10, 60)])

        hour = insert('hour: "') + hour_padded + insert('"')
        oclock = cross("o'clock", "") | cross("oclock", "") | cross("hundred hours", "")
        minute = (
            oclock + insert("00")
            | delete("o") + ds + graph_min_single
            | graph_min_double
        )

        suffix = ds + insert(' noon: "') + time_suffix + insert('"')
        zone = ds + insert(' zone: "') + time_zone + insert('"')

        # "eight oclock" (no suffix needed)
        graph_oclock = hour + ds + insert(' minute: "') + oclock + insert('00"')
        # "two o five" (no suffix needed)
        graph_o_min = hour + ds + insert(' minute: "') + delete("o") + ds + graph_min_single + insert('"')
        # "two pm", "three am" (hour + suffix, minutes = 00)
        graph_h_suffix = hour + insert(' minute: "00"') + suffix + closure(zone, 0, 1)
        # "two thirty am" (hour + minute + suffix required)
        graph_hm_suffix = (
            hour + ds + insert(' minute: "') + graph_min_double + insert('"')
            + suffix + closure(zone, 0, 1)
        )
        # "half past two", "quarter past two"
        graph_half_quarter = (
            insert('minute: "')
            + (cross("half", "30") | cross("quarter", "15"))
            + insert('"')
            + ds + delete("past") + ds
            + hour
        )

        final_graph = graph_oclock | graph_o_min | graph_h_suffix | graph_hm_suffix | graph_half_quarter
        self.tagger = self.add_tokens(final_graph)

    def build_verbalizer(self):
        hour = delete('hour: "') + self.NOT_QUOTE.plus + delete('"')
        minute = delete(' minute: "') + self.NOT_QUOTE.plus + delete('"')
        noon = delete(' noon: "') + self.NOT_QUOTE.plus + delete('"')
        graph = hour + insert(":") + self.DELETE_SPACE + minute
        graph += closure(insert(" ") + self.DELETE_SPACE + noon, 0, 1)
        self.verbalizer = self.delete_tokens(graph)
