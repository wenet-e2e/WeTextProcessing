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
from pynini.lib.pynutil import add_weight, delete, insert

from itn.utils import get_abs_path
from tn.processor import Processor


TOKEN_ORDER = {"time": ["prefix", "to", "hours", "minutes", "suffix"]}


class Time(Processor):
    """Raw spoken clock times.

    "às três horas e trinta minutos" -> às 3:30. With a "para" phrase the
    minutes count down to the next hour: "às quinze pras duas" keeps
    minutes "quinze", to "pras" and hours "duas" and verbalizes as às 1:45.
    """

    def __init__(self, cardinal):
        super().__init__("time", ordertype="itn", token_orders=TOKEN_ORDER)

        def time_file(name):
            return pynini.string_file(get_abs_path("portuguese/data/time/" + name))

        hours_to = time_file("hours_to.tsv")
        minutes_to = time_file("minutes_to.tsv")
        suffix = (time_file("time_suffix_am.tsv") | time_file("time_suffix_pm.tsv")).project("input")
        one_to_hundred = pynini.union(
            cardinal.digit,
            cardinal.twenties,
            cardinal.teen,
            cardinal.ties + insert("0"),
            cardinal.ties + delete(" e ") + cardinal.digit,
        )
        two_to_23 = (one_to_hundred @ pynini.union(*[str(h) for h in range(2, 24)])).optimize()
        one_to_59 = (one_to_hundred @ pynini.union(*[str(m) for m in range(1, 60)])).optimize()
        one = pynini.cross("uma", "1")
        noon = pynini.cross("meio dia", "12")
        midnight = pynini.cross("meia noite", "0")
        half = pynini.cross("meia", "30")
        half_noon = pynini.cross("meio", "30")

        space = self.DELETE_SPACE
        gap = delete(self.SPACE.plus) + insert(" ")
        connector = space + delete(" e ") + space
        hour_word = space + delete(pynini.accep("hora") + pynini.accep("s").ques)
        minute_word = space + delete(pynini.accep("minuto") + pynini.accep("s").ques)

        def field(name, graph):
            return self.tag_field(name, graph)

        prefix_one = pynini.union("à", "a")
        prefix_many = pynini.union("às", "as")
        hours_one = field("prefix", prefix_one) + gap + field("hours", one)
        hours_many = field("prefix", prefix_many) + gap + field("hours", two_to_23)
        minutes = (
            connector + insert(" ") + field("minutes", one_to_59) + minute_word.ques
            | connector + insert(" ") + field("minutes", half) + hour_word.ques
        )
        clock = (hours_one | hours_many) + hour_word.ques + minutes
        clock |= add_weight((hours_one | hours_many) + hour_word, 0.2)
        clock |= field("hours", noon | midnight) + minutes
        clock |= field("hours", noon) + connector + insert(" ") + field("minutes", half_noon)

        to_many = pynini.union("para", "para a", "para as", "pra", "pras")
        to_noon = pynini.union("para", "para o", "pro")
        countdown = (one_to_59 @ minutes_to).optimize()
        to_prefix = field("prefix", prefix_one | prefix_many) + gap + field("minutes", countdown) + minute_word.ques
        to_hours = (two_to_23 @ hours_to | midnight @ hours_to).optimize()
        countdown_clock = (
            to_prefix + gap + field("to", to_many) + gap + field("hours", to_hours) + hour_word.ques
            | to_prefix + gap + field("to", to_noon) + gap + field("hours", noon @ hours_to)
        )
        suffix_t = (space + insert(" ") + field("suffix", suffix)).ques
        self.tagger = self.add_tokens((clock | countdown_clock) + suffix_t)

        pad = self.DIGIT**2 | insert("0") + self.DIGIT
        prefix_v = self.verbalize_field("prefix", pynini.cross("a", "à") | "à" | pynini.cross("as", "às") | "às")
        prefix_v = (prefix_v + delete(" ") + insert(" ")).ques
        hours = one | two_to_23 | noon | midnight
        minutes_v = (one_to_59 | half | half_noon) @ pad
        suffix_v = (insert(" ") + self.verbalize_field("suffix", suffix, leading_space=True)).ques
        plain = (
            self.verbalize_field("hours", hours) + insert(":")
            + (self.verbalize_field("minutes", minutes_v, leading_space=True) | insert("00"))
        )
        counted = (
            self.verbalize_field("to", delete(to_many | to_noon)) + delete(" ")
            + self.verbalize_field("hours", to_hours | noon @ hours_to) + insert(":")
            + self.verbalize_field("minutes", countdown @ pad, leading_space=True)
        )
        self.verbalizer = self.delete_tokens(prefix_v + (plain | counted) + suffix_v)
