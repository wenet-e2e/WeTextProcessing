# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

from itn.armenian.rules.utils import preferred_map
from tn.processor import Processor


class Time(Processor):
    """Spoken clock times.

    ``տասներկուսն անց հինգ`` -> ``12:05`` (hours past), ``հինգին տասնհինգ
    պակաս`` -> ``04:45`` (minutes to the hour) and ``տասներեք անց կես`` ->
    ``13:30``. The tagger keeps the spoken hours and minutes and the
    ``relation`` word (``անց`` or ``պակաս``) that selects the arithmetic.
    """

    def __init__(self):
        super().__init__(name="time", ordertype="itn", token_orders={})
        ending = (delete("ն") | delete("ին")).ques
        hours = preferred_map("armenian/data/time/hours.tsv") + ending
        minutes = preferred_map("armenian/data/time/minutes.tsv") + ending
        past_minutes = minutes | pynini.cross("կես", "30")
        to_hours = (hours + delete("ին")) @ preferred_map("armenian/data/time/to_hour.tsv")
        to_minutes = minutes @ preferred_map("armenian/data/time/minutes_to.tsv")
        pad = self.DIGIT**2 | insert("0") + self.DIGIT
        hours = (hours @ pad).optimize()
        past_minutes = (past_minutes @ pad).optimize()
        to_hours = (to_hours @ pad).optimize()
        to_minutes = to_minutes.optimize()

        separator = delete(self.SPACE.plus) + insert(" ")
        past = self.tag_field("hours", hours) + separator + self.tag_field("relation", pynini.accep("անց"))
        past += separator + self.tag_field("minutes", past_minutes)
        to = self.tag_field("hours", to_hours) + separator + self.tag_field("minutes", to_minutes)
        to += separator + self.tag_field("relation", pynini.accep("պակաս"))
        self.tagger = self.add_tokens(past | to)

        past = self.verbalize_field("hours", hours)
        past += self.verbalize_field("relation", delete("անց"), leading_space=True) + insert(":")
        past += self.verbalize_field("minutes", past_minutes, leading_space=True)
        to = self.verbalize_field("hours", to_hours) + insert(":")
        to += self.verbalize_field("minutes", to_minutes, leading_space=True)
        to += self.verbalize_field("relation", delete("պակաս"), leading_space=True)
        self.verbalizer = self.delete_tokens(past | to)
