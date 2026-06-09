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

from pynini import closure, cross, string_file
from pynini.lib.pynutil import delete, insert

from itn.english.rules.cardinal import Cardinal
from tn.processor import Processor
from tn.utils import get_abs_path


class Time(Processor):

    def __init__(self, cardinal=None):
        super().__init__(name="time", ordertype="itn")
        self.cardinal = cardinal or Cardinal()
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        digit = string_file(get_abs_path("../itn/english/data/numbers/digit.tsv"))
        teen = string_file(get_abs_path("../itn/english/data/numbers/teen.tsv"))
        ties = string_file(get_abs_path("../itn/english/data/numbers/ties.tsv"))
        time_suffix = string_file(get_abs_path("../itn/english/data/time/time_suffix.tsv"))
        time_zone = string_file(get_abs_path("../itn/english/data/time/time_zone.tsv"))
        ds = delete(" ")

        hour = teen | (insert("0") + digit)
        minute = teen | (ties + (ds + digit | insert("0"))) | insert("0") + digit

        # two thirty => 02:30
        graph = insert('hour: "') + hour + insert('" ') + ds + insert('minute: "') + minute + insert('"')
        # eight oclock => 08:00
        oclock = cross("o'clock", "") | cross("oclock", "")
        graph |= insert('hour: "') + hour + insert('" minute: "00"') + ds + oclock

        suffix = ds + insert(' noon: "') + time_suffix + insert('"')
        zone = ds + insert(' zone: "') + time_zone + insert('"')
        graph += suffix.ques + zone.ques

        self.tagger = self.add_tokens(graph)

    def build_verbalizer(self):
        hours = delete('hour: "') + self.NOT_QUOTE.plus + delete('"')
        minutes = delete(' minute: "') + self.NOT_QUOTE.plus + delete('"')
        suffix = delete(' noon: "') + self.NOT_QUOTE.plus + delete('"')
        zone = delete(' zone: "') + self.NOT_QUOTE.plus + delete('"')
        graph = hours + insert(":") + self.DELETE_SPACE + minutes
        graph += closure(insert(" ") + self.DELETE_SPACE + suffix, 0, 1)
        graph += closure(insert(" ") + self.DELETE_SPACE + zone, 0, 1)
        self.verbalizer = self.delete_tokens(graph)
