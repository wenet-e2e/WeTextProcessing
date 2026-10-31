# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from pynini.lib.pynutil import add_weight, delete, insert

from tn.hindi.rules.utils import HI_DEDH, HI_DHAI, HI_PAUNE, HI_SADHE, HI_SAVVA, hindi_path
from tn.processor import Processor

TOKEN_ORDER = {}


class Time(Processor):
    """``१२:३०:३०`` -> ``time { hours: "१२" minutes: "३०" seconds: "३०" }``.

    Quarter-hour idioms (``1:30`` -> डेढ़) keep the whole clock as one raw ``value``.
    """

    def __init__(self, cardinal):
        super().__init__("time", token_orders=TOKEN_ORDER)
        hours = pynini.string_file(hindi_path("time/hours.tsv"))
        minutes = pynini.string_file(hindi_path("time/minutes.tsv"))
        seconds = pynini.string_file(hindi_path("time/seconds.tsv"))
        double_zero = pynini.union("००", "00")
        small = cardinal.digit | cardinal.teens_and_ties
        paune = pynini.string_file(hindi_path("whitelist/paune_mappings.tsv"))

        colon = delete(":") + insert(" ")
        hour_tag = self.tag_field("hours", hours)
        minute_tag = self.tag_field("minutes", minutes)
        hms = hour_tag + colon + minute_tag + colon + self.tag_field("seconds", seconds)
        hm = add_weight(hour_tag + colon + minute_tag, 0.3)
        hour_only = add_weight(hour_tag + colon + self.tag_field("minutes", double_zero), 0.3)

        idioms = pynini.union(
            add_weight(pynini.string_map([(h + ":" + m, HI_DEDH) for h in ("१", "1") for m in ("३०", "30")]
                                         + [(h + ":" + m, HI_DHAI) for h in ("२", "2") for m in ("३०", "30")]), 0.1),
            add_weight(insert(HI_SAVVA + " ") + small + delete(pynini.union(":१५", ":15")), 0.2),
            add_weight(insert(HI_SADHE + " ") + small + delete(pynini.union(":३०", ":30")), 0.2),
            add_weight(insert(HI_PAUNE + " ") + paune + delete(pynini.union(":४५", ":45")), 0.1),
        ).optimize()
        self.tagger = self.add_tokens(hms | hm | hour_only | self.tag_field("value", idioms))

        hour = self.verbalize_field("hours", hours)
        minute = insert(" ") + self.verbalize_field("minutes", minutes, leading_space=True)
        second = insert(" ") + self.verbalize_field("seconds", seconds, leading_space=True)
        clock = hour + insert(" बजकर") + minute + insert(" मिनट")
        hour_only = hour + self.verbalize_field("minutes", delete(double_zero), leading_space=True) + insert(" बजे")
        self.verbalizer = self.delete_tokens(
            clock + second + insert(" सेकंड") | clock | hour_only | self.verbalize_field("value", idioms))
