# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pynini.lib import pynutil

from itn.swedish.rules.date import pad
from itn.swedish.rules.utils import table
from tn.processor import Processor

TOKEN_ORDER = {"time": ["prefix", "hours", "minutes", "seconds", "suffix", "zone"]}

QUARTERS = {15: "kvart över", 30: "halv", 45: "kvart i"}


def values(start, stop):
    return pynini.union(*[str(x) for x in range(start, stop)])


class Time(Processor):
    """Raw spoken clock times, including ``kvart i tolv`` phrases."""

    def __init__(self, cardinal):
        super().__init__(name="time", ordertype="itn", token_orders=TOKEN_ORDER)
        number = cardinal.graph
        suffix = table("time/suffix")
        zone = table("time/time_zone")
        prefix = pynini.cross(pynini.union("klockan", "klockan är"), "kl.")
        hours = (number @ values(0, 24)).optimize()
        minutes = (number @ values(1, 60)).optimize()
        # "kvart i tolv" names the next hour.
        previous = pynini.string_map([(str(x % 12 + 1), str(x)) for x in range(1, 13)])
        hours_to = (number @ previous).optimize()

        def shifted(pairs):
            return (number @ pynini.string_map(pairs)).optimize()

        to_phrase = pynini.string_map([(word, str(value)) for value, word in QUARTERS.items() if value != 15])
        from_phrase = pynini.cross("kvart över", "15")
        for value, word in QUARTERS.items():
            after = shifted([(str(x), str(value + x)) for x in range(1, 60 - value)])
            before = shifted([(str(x), str(value - x)) for x in range(1, value)])
            phrases = (
                after + pynutil.delete(" över " + word)
                | before + pynutil.delete(" i " + word)
            )
            if value == 15:
                from_phrase |= phrases
            else:
                to_phrase |= phrases
        to_phrase |= shifted([(str(x), str(60 - x)) for x in range(1, 60)]) + pynutil.delete(" i")
        from_phrase |= minutes + pynutil.delete(" över")
        to_phrase = to_phrase.optimize()
        from_phrase = from_phrase.optimize()

        space = pynini.accep(" ")
        prefix_t = self.tag_field("prefix", prefix) + space
        hours_t = self.tag_field("hours", hours)
        hours_any_t = (prefix_t.ques + hours_t)
        minutes_t = space + self.tag_field("minutes", minutes)
        seconds_t = space + self.tag_field("seconds", minutes)
        suffix_t = space + self.tag_field("suffix", suffix)
        zone_t = space + self.tag_field("zone", zone)
        some_suffix = suffix_t + zone_t.ques | suffix_t.ques + zone_t
        tagger = (
            self.tag_field("minutes", to_phrase) + space + self.tag_field("hours", hours_to)
            | self.tag_field("minutes", from_phrase) + space + hours_any_t
            | prefix_t + hours_t + suffix_t.ques + zone_t.ques
            | hours_any_t + (minutes_t + seconds_t.ques).ques + some_suffix
        )
        self.tagger = self.add_tokens(tagger)

        def field(name, graph):
            return pynutil.delete(" ") + self.verbalize_field(name, graph)

        colon = pynutil.insert(":")
        prefix_v = self.verbalize_field("prefix", prefix) + pynutil.insert(" ") + field("hours", hours)
        hours_v = self.verbalize_field("hours", hours)
        tail = (pynutil.insert(" ") + field("suffix", suffix)).ques + (pynutil.insert(" ") + field("zone", zone)).ques
        clock = colon + field("minutes", pad(minutes)) + (colon + field("seconds", pad(minutes))).ques
        verbalizer = (
            (prefix_v | self.verbalize_field("hours", pad(hours))) + clock + tail
            | (prefix_v | hours_v) + tail
            | self.verbalize_field("hours", pad(hours_to)) + colon + field("minutes", pad(to_phrase))
            | (prefix_v | self.verbalize_field("hours", pad(hours))) + colon + field("minutes", pad(from_phrase))
        )
        self.verbalizer = self.delete_tokens(verbalizer)
