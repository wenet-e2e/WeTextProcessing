# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from itn.french.rules.cardinal import load
from itn.french.rules.decimal import WHITE_SPACE
from tn.processor import Processor


TOKEN_ORDER = {"time": ["hours", "minutes", "suffix"]}


class Time(Processor):
    """``midi moins le quart`` -> ``time { hours: "midi" minutes: "moins le quart" }`` -> ``11 h 45``.

    ``du soir`` / ``de l'après-midi`` move the hour to the afternoon.
    """

    def __init__(self):
        super().__init__("time", ordertype="itn", token_orders=TOKEN_ORDER)
        delete_space = delete(WHITE_SPACE.star)
        heures = delete(pynini.accep("heure") + pynini.accep("s").ques)
        hours = load("time/hours") + delete_space + heures | pynini.cross("midi", "12") | pynini.cross("minuit", "0")
        hours = hours.optimize()
        minutes = load("time/minutes")
        demi = pynini.cross(pynini.accep("demi") + pynini.accep("e").ques, "30")
        quart = pynini.cross(pynini.accep("le ").ques + "quart", "15")
        fractions = demi | quart | pynini.cross("trois quarts", "45")

        standard = (hours, (minutes | delete("et") + delete_space + fractions).optimize(), True)
        until = (
            (hours @ load("time/hours_to")).optimize(),
            (delete("moins") + delete_space + ((minutes | fractions) @ load("time/minutes_to"))).optimize(),
            False,
        )
        night = load("time/hour_to_night")
        am = pynini.cross(load("time/time_suffix_am"), "")
        pm = pynini.cross(load("time/time_suffix_pm"), "")

        taggers, verbalizers = [], []
        for hour, minute, optional in (standard, until):
            for suffix, shift in ((None, False), (am, False), (pm, True)):
                shifted = (hour @ night).optimize() if shift else hour
                tagger = self.tag_field("hours", shifted)
                verbalizer = self.verbalize_field("hours", shifted) + insert(" h")
                minute_tagger = delete_space + insert(" ") + self.tag_field("minutes", minute)
                minute_verbalizer = pynini.accep(" ") + self.verbalize_field("minutes", minute)
                tagger += minute_tagger.ques if optional else minute_tagger
                verbalizer += minute_verbalizer.ques if optional else minute_verbalizer
                if suffix is not None:
                    tagger += delete_space + insert(" ") + self.tag_field("suffix", suffix)
                    verbalizer += self.verbalize_field("suffix", suffix, leading_space=True)
                taggers.append(tagger)
                verbalizers.append(verbalizer)
        self.tagger = self.add_tokens(pynini.union(*taggers))
        self.verbalizer = self.delete_tokens(pynini.union(*verbalizers))
