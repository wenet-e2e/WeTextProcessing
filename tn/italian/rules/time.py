# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from tn.processor import Processor

# Penalty for the idiomatic ``un quarto``/``mezza`` readings, kept for n-best.
IDIOM_WEIGHT = 0.01


class Time(Processor):
    """Clock times, e.g. ``17:15:26`` -> ``diciassette e quindici minuti e ventisei secondi``.

    The tagger keeps the written hour, minute and second digits; the
    verbalizer reads them, dropping ``:00`` minutes when there are no
    seconds.
    """

    def __init__(self, cardinal):
        super().__init__("time", token_orders={})
        digit = self.DIGIT
        number = pynini.cross("0", "zero") | cardinal.digits | cardinal.two_digits
        hour_digits = pynini.union(*[str(hour) for hour in range(25)])
        hours = ((delete("0") + digit | (digit - "0") + digit | digit) @ hour_digits @ number).optimize()
        minute_value = (delete("0") + digit | pynini.union(*"12345") + digit) @ number
        two = pynini.closure(digit, 2, 2)
        minutes = pynini.cross("01", "un minuto") | (two - "01") @ minute_value + insert(" minuti")
        minutes |= add_weight(pynini.cross("15", "un quarto") | pynini.cross("30", "mezza"), IDIOM_WEIGHT)
        minutes = minutes.optimize()
        seconds = pynini.cross("01", "un secondo") | (two - "01") @ minute_value + insert(" secondi")
        seconds = seconds.optimize()

        tagger = self.tag_field("hours", hours) + delete(":") + insert(" ")
        tagger += self.tag_field("minutes", minutes)
        tagger += (delete(":") + insert(" ") + self.tag_field("seconds", seconds)).ques
        self.tagger = self.add_tokens(tagger)

        hours_verb = self.verbalize_field("hours", hours)

        def and_field(name, graph):
            return insert(" e ") + self.verbalize_field(name, graph, leading_space=True)

        no_minutes = delete(" ") + self.verbalize_field("minutes", pynini.cross("00", ""))
        minutes_only = and_field("minutes", (two - "00") @ minutes)
        with_seconds = and_field("minutes", minutes) + and_field("seconds", seconds)
        verbalizer = hours_verb + (no_minutes | minutes_only | with_seconds)
        self.verbalizer = self.delete_tokens(verbalizer)
