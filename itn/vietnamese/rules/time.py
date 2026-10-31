# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
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

from itn.utils import get_abs_path
from itn.vietnamese.rules.cardinal import DELETE_EXTRA_SPACE, DELETE_SPACE
from tn.processor import Processor


def time_file(name):
    return pynini.string_file(get_abs_path("vietnamese/data/time/" + name + ".tsv")).optimize()


class Time(Processor):
    """Spoken clock times keep their words raw: ``chín giờ kém mười`` -> ``8:50``.

    ``kém`` ("to") is kept as a ``relation`` field after the hour; the
    verbalizer then reads the hour and minutes backwards from the next hour.
    """

    def __init__(self):
        super().__init__(name="time", ordertype="itn", token_orders={})
        ds = DELETE_SPACE
        sep = DELETE_EXTRA_SPACE + insert(" ")
        self.hour_graph = time_file("hours")
        minutes = time_file("minutes")
        self.minute_graph = (minutes | time_file("zero_prefix") + ds + minutes).optimize()
        half = pynini.cross("rưỡi", "30")
        self.zone_graph = time_file("time_zone").invert().optimize()
        self.hour_to_graph = (self.hour_graph @ time_file("hours_to")).optimize()
        self.minute_to_graph = (minutes @ time_file("minutes_to")).optimize()

        hour = self.tag_field("hours", self.hour_graph) + ds + delete("giờ")
        minute_word = ds + delete("phút")
        second = self.tag_field("seconds", self.minute_graph) + ds + delete("giây")
        plain_minute = self.tag_field("minutes", minutes) + minute_word
        to_hour = self.tag_field("hours", self.hour_to_graph) + ds + delete("giờ") + ds + insert(" ")
        to_hour += self.tag_field("relation", pynini.accep("kém"))
        zone = (ds + insert(" ") + self.tag_field("zone", self.zone_graph)).ques
        self.tagger = self.add_tokens(pynini.union(
            hour,
            hour + sep + (self.tag_field("minutes", self.minute_graph) + minute_word.ques
                          | self.tag_field("minutes", half)),
            hour + sep + plain_minute + sep + second,
            # NeMo's hour-second form needs two separate runs of whitespace.
            hour + DELETE_EXTRA_SPACE + sep + second,
            plain_minute + sep + (second | self.tag_field("seconds", half)),
            to_hour + sep + self.tag_field("minutes", self.minute_to_graph) + minute_word.ques,
        ) + zone)

        pad = self.DIGIT**2 | insert("0") + self.DIGIT
        minute_graph = self.minute_graph | half
        hour_v = self.verbalize_field("hours", self.hour_graph)
        minute_v = insert(":") + delete(" ") + self.verbalize_field("minutes", minute_graph @ pad)
        second_v = insert(":") + delete(" ") + self.verbalize_field("seconds", minute_graph @ pad)
        to_v = self.verbalize_field("hours", self.hour_to_graph) + delete(" ")
        to_v += self.verbalize_field("relation", delete("kém")) + insert(":") + delete(" ")
        to_v += self.verbalize_field("minutes", self.minute_to_graph @ pad)
        minute_second_v = self.verbalize_field("minutes", minutes) + insert("p") + delete(" ")
        minute_second_v += self.verbalize_field("seconds", minute_graph @ pad) + insert("s")
        zone_v = (insert(" ") + self.verbalize_field("zone", self.zone_graph, True)).ques
        self.verbalizer = self.delete_tokens(pynini.union(
            hour_v + insert("h"),
            hour_v + minute_v + second_v.ques,
            hour_v + insert(":00") + second_v,
            to_v,
            minute_second_v,
        ) + zone_v)
