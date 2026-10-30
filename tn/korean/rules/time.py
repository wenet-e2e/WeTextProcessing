# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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


TOKEN_ORDER = {"time": ["suffix", "hours", "minutes", "seconds"]}


class Time(Processor):

    def __init__(self, cardinal):
        super().__init__("time", token_orders=TOKEN_ORDER)
        strip_zero = delete("0").ques
        native_hour = pynini.string_file(get_abs_path("korean/data/time/hour.tsv"))
        sino_hour = pynini.union("0", "00", *[str(i) for i in range(13, 25)])
        hour = (strip_zero + native_hour | sino_hour @ (strip_zero + cardinal.graph)).optimize()
        self.hours = (hour + insert("시") | hour + pynini.accep("시")
                      | pynini.string_map([("정오", "열두시"), ("자정", "영시")])).optimize()
        self.suffix = pynini.string_file(get_abs_path("korean/data/time/division.tsv")).optimize()

        number = (strip_zero + cardinal.graph).optimize()
        zero = pynini.union("0", "00")
        nonzero = (pynini.difference(self.input_projection(number), zero) @ number).optimize()
        colon = delete(":") + delete(self.SPACE).ques
        self.minutes = (nonzero + "분" | delete(zero + "분")
                        | pynini.accep("반") + pynini.union("쯤", "경").ques
                        | colon + nonzero + insert("분") | colon + delete(zero)).optimize()
        self.seconds = (nonzero + "초" | delete(zero + "초")
                        | colon + nonzero + insert("초") | colon + delete(zero)).optimize()
        self.leading_minutes = self._leading_component(self.minutes)
        self.leading_seconds = self._leading_component(self.seconds)

        field_separator = self.DELETE_SPACE + insert(" ")
        hours_ko = self.tag_field("hours", hour + "시")
        minutes_ko = self.tag_field("minutes", number + "분" | pynini.accep("반")
                                    + pynini.union("쯤", "경").ques)
        seconds_ko = self.tag_field("seconds", number + "초")
        following = field_separator + minutes_ko + (field_separator + seconds_ko).ques
        hangul = (hours_ko + following.ques
                  | minutes_ko + (field_separator + seconds_ko).ques | seconds_ko)
        clock = self.tag_field("hours", hour) + field_separator
        clock += self.tag_field("minutes", colon + number)
        clock += (field_separator + self.tag_field("seconds", colon + number)).ques
        special = self.tag_field("hours", pynini.union("정오", "자정"))
        body = (hangul | clock | special).optimize()
        suffix = self.tag_field("suffix", self.suffix)
        self.tagger = self.add_tokens(body | suffix + field_separator + body | body + field_separator + suffix)

        hours = self.verbalize_field("hours", self.hours)
        minutes = self.verbalize_field("minutes", self.minutes)
        seconds = self.verbalize_field("seconds", self.seconds)
        following_minutes = self.verbalize_field("minutes", self.leading_minutes, leading_space=True)
        following_seconds = self.verbalize_field("seconds", self.leading_seconds, leading_space=True)
        spoken = (hours + following_minutes.ques + following_seconds.ques
                  | minutes + following_seconds.ques | seconds)
        prefix = self.verbalize_field("suffix", self.suffix) + delete(" ") + insert(" ")
        self.verbalizer = self.delete_tokens(prefix.ques + spoken)

    def _leading_component(self, semantic):
        # A clock's :00 is silent, including its separator, but remains tagged.
        # Explicit Korean zero minutes/seconds retain the source's output space.
        clock_zero = pynini.accep(":") + self.SPACE.ques + "00"
        other = pynini.difference(self.input_projection(semantic), clock_zero) @ semantic
        return (delete(clock_zero) | insert(" ") + other).optimize()
