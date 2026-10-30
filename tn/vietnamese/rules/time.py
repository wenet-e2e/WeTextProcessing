# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from tn.utils import get_abs_path


TOKEN_ORDER = {"time": ["hours", "minutes", "seconds", "zone"]}


class Time(Processor):
    """Clock times and durations (``8:05``, ``14h30``, ``3p20s``, ``2 giờ 15 phút``) with raw fields."""

    def __init__(self, cardinal):
        super().__init__("time", token_orders=TOKEN_ORDER)
        canonical = (delete("0").ques | (self.DIGIT - "0")) + self.DIGIT
        hours = pynini.union(*[str(x) for x in range(25)])
        sixty = pynini.union(*[str(x) for x in range(60)])
        self.hour_graph = (canonical @ hours @ cardinal.graph).optimize()
        self.sixty_graph = (canonical @ sixty @ cardinal.graph).optimize()
        self.zone_graph = pynini.string_file(get_abs_path("vietnamese/data/time/time_zones.tsv")).optimize()

        hour = self.tag_field("hours", self.hour_graph)
        minute = self.tag_field("minutes", self.sixty_graph)
        second = self.tag_field("seconds", self.sixty_graph)
        zone = self.tag_field("zone", self.zone_graph)
        sep = insert(" ")
        spaced_zone = (pynini.accep(" ") + zone).ques
        suffix = delete(pynini.union("h", "g"))
        colon = delete(":") + sep
        hour_word = delete(" giờ")
        minute_word = delete(" phút")
        second_word = delete(" giây")
        word_sep = delete(" ") + sep
        self.fields_tagger = pynini.union(
            hour + colon + minute + spaced_zone,
            hour + colon + minute + colon + second + spaced_zone,
            hour + suffix + spaced_zone,
            hour + suffix + sep + minute + (sep + zone).ques,
            hour + suffix + sep + zone,
            minute + delete("p"),
            second + delete("s"),
            minute + delete("p") + sep + second + delete("s"),
            hour + hour_word + spaced_zone,
            hour + hour_word + word_sep + minute + minute_word + spaced_zone,
            hour + hour_word + word_sep + minute + minute_word + word_sep + second + second_word + spaced_zone,
            minute + minute_word,
            minute + minute_word + word_sep + second + second_word,
            second + second_word,
        ).optimize()
        self.tagger = self.add_tokens(self.fields_tagger)

        # NeMo drops "không phút" / "không giây" unless it is the first spoken component.
        zero = self.input_projection(canonical @ "0")
        nonzero = pynini.difference(self.input_projection(self.sixty_graph), zero)

        def first(name, unit):
            return self.verbalize_field(name, self.sixty_graph) + insert(" " + unit)

        def later(name, unit):
            spoken = insert(" ") + self.verbalize_field(name, nonzero @ self.sixty_graph, True) + insert(" " + unit)
            return spoken | self.verbalize_field(name, delete(zero), True)

        hour_v = self.verbalize_field("hours", self.hour_graph) + insert(" giờ")
        zone_v = insert(" ") + self.verbalize_field("zone", self.zone_graph, True)
        self.fields_verbalizer = pynini.union(
            hour_v + later("minutes", "phút").ques + later("seconds", "giây").ques + zone_v.ques,
            first("minutes", "phút") + later("seconds", "giây").ques,
            first("seconds", "giây"),
        ).optimize()
        self.verbalizer = self.delete_tokens(self.fields_verbalizer)
