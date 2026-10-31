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

from tn.processor import Processor
from tn.swedish.rules.utils import alt, labels, tsv

TOKEN_ORDER = {"time": ["prefix", "hours", "minutes", "seconds", "suffix", "zone"]}


class Time(Processor):
    """Raw clock digits, ``kl.`` prefixes, day-period suffixes and zones."""

    def __init__(self, cardinal, deterministic=True):
        super().__init__("time", token_orders=TOKEN_ORDER)
        self.deterministic = deterministic
        digit = self.DIGIT
        number = cardinal.graph
        suffix = pynini.string_map(labels("time/suffix")).optimize()
        zone = tsv("time/time_zone").optimize()
        prefix = pynini.union(
            pynini.cross("kl.", "klockan"), pynini.cross("kl", "klockan"), "klockan", "klockan är")

        def numbers(start, stop):
            return pynini.union(*[str(x) for x in range(start, stop)]) @ number

        hours = ((digit + digit | pynutil.delete("0").ques + digit) @ numbers(0, 24)).optimize()
        single = pynutil.delete("0") + numbers(1, 10)
        minutes = single | numbers(10, 60)
        seconds = minutes | pynini.cross("00", "noll")
        if not deterministic:
            minutes |= alt(pynini.cross("00", "noll noll"))
            seconds |= alt(pynini.cross("0", "noll ") + numbers(1, 10)) | alt(pynini.cross("00", "noll noll"))
        minutes = minutes.optimize()
        seconds = seconds.optimize()

        ins = pynutil.insert(" ")
        ensure_space = pynutil.delete(" ").ques + ins
        hour_sep = pynutil.delete(pynini.union(":", ".")) + ins

        prefix_t = self.tag_field("prefix", prefix) + ensure_space
        hours_t = self.tag_field("hours", hours)
        zero_minutes = pynini.accep("00")
        minutes_t = self.tag_field("minutes", minutes | zero_minutes)
        seconds_t = self.tag_field("seconds", seconds)
        suffix_t = self.tag_field("suffix", suffix)
        zone_t = pynini.accep(" ") + self.tag_field("zone", zone)
        suffix_opt_t = (ensure_space + suffix_t).ques
        zone_opt_t = zone_t.ques
        required = ensure_space + (suffix_t + zone_opt_t | self.tag_field("zone", zone))

        def clock(seps):
            graph = hours_t + seps[0] + minutes_t
            if len(seps) > 1:
                graph += seps[1] + seconds_t
            return graph

        colon = pynutil.delete(":") + ins
        point = pynutil.delete(".") + ins
        hm = clock([hour_sep])
        hms = clock([colon, colon]) | clock([point, point])
        tagger = (
            prefix_t + (hm | hms) + suffix_opt_t + zone_opt_t
            | (hm | hms) + required
            | prefix_t + hours_t + suffix_opt_t + zone_opt_t
            | hours_t + required
        )
        if not deterministic:
            tagger |= hms
        self.tagger = self.add_tokens(tagger)

        minutes_v = (
            self.verbalize_field("minutes", pynutil.delete(zero_minutes), leading_space=True)
            | ins + self.verbalize_field("minutes", minutes, leading_space=True)
        )
        verbalizer = (
            (self.verbalize_field("prefix", prefix) + ins + self.verbalize_field("hours", hours, leading_space=True)
             | self.verbalize_field("hours", hours))
            + minutes_v.ques
            + (ins + self.verbalize_field("seconds", seconds, leading_space=True)).ques
            + (ins + self.verbalize_field("suffix", suffix, leading_space=True)).ques
            + (ins + self.verbalize_field("zone", zone, leading_space=True)).ques
        )
        self.verbalizer = self.delete_tokens(verbalizer)
