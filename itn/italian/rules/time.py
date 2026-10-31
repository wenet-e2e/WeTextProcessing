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

from itn.italian.rules.cardinal import two_digit_pairs
from itn.italian.rules.utils import string_map
from tn.processor import Processor

PREPOSITIONS = ("alle", "dalle")
IDIOMS = {"un quarto": 15, "mezza": 30, "mezzo": 30, "tre quarti": 45}


class Time(Processor):
    """Clock times, e.g. ``cinque e un quarto`` -> ``05:15``.

    The tagger keeps the spoken hour, minutes (without ``minuti``) and
    seconds (without ``secondi``), the ``meno`` relation and a leading
    ``alle``/``dalle``, which also licenses a bare hour (``alle sedici`` ->
    ``alle 16:00``); the verbalizer writes ``HH:MM[:SS]``.
    """

    def __init__(self):
        super().__init__("time", token_orders={})
        spoken_numbers = {}
        for written, spoken in two_digit_pairs():
            spoken_numbers.setdefault(int(written), []).append(spoken)
        hour_pairs = [("00", "zero"), ("01", "una")]
        hour_pairs += [("{:02d}".format(h), s) for h in range(1, 25) for s in spoken_numbers[h]]
        hours = string_map(hour_pairs)
        previous = string_map([("{:02d}".format(int(w) - 1), s) for w, s in hour_pairs if int(w) > 0])
        special = string_map([("12", "mezzogiorno"), ("00", "mezzanotte")])

        def counted(unit, plural):
            pairs = [("01", "un " + unit)]
            pairs += [("{:02d}".format(m), s + " " + plural) for m in range(2, 60) for s in spoken_numbers[m]]
            return pairs

        minute_pairs = counted("minuto", "minuti") + [("{:02d}".format(v), s) for s, v in IDIOMS.items()]
        minutes = string_map(minute_pairs)
        # ``meno un quarto`` / ``meno dieci minuti`` count back from the hour.
        before = string_map([
            ("{:02d}".format(60 - int(w)), s) for w, s in minute_pairs if s not in ("mezza", "mezzo", "tre quarti")
        ])
        bare_minutes = string_map([("{:02d}".format(m), s) for m in range(1, 60) for s in spoken_numbers[m]])
        seconds = string_map(counted("secondo", "secondi"))

        space = pynini.accep(" ")
        preposition = pynini.union(*PREPOSITIONS)
        prep_tag = self.tag_field("preposition", preposition) + space
        hours_tag = self.tag_field("hours", hours)
        hours_prev_tag = self.tag_field("hours", previous)
        and_ = delete(" e") + space
        seconds_tag = and_ + self.tag_field("seconds", seconds)
        clock = hours_tag + and_ + self.tag_field("minutes", minutes)
        clock |= hours_prev_tag + space + self.tag_field("relation", pynini.accep("meno")) + space + self.tag_field("minutes", before)
        clock += seconds_tag.ques
        loose = prep_tag + hours_tag + (and_ + self.tag_field("minutes", bare_minutes)).ques
        noon = prep_tag.ques + self.tag_field("hours", special)
        self.tagger = self.add_tokens(prep_tag.ques + clock | loose | noon)

        prep_verb = self.verbalize_field("preposition", preposition) + space
        colon = delete(" ") + insert(":")
        hours_verb = self.verbalize_field("hours", hours)
        seconds_verb = colon + self.verbalize_field("seconds", seconds)
        clock_verb = hours_verb + colon + self.verbalize_field("minutes", minutes)
        clock_verb |= (self.verbalize_field("hours", previous) + delete(" ") + self.verbalize_field("relation", delete("meno"))
                       + colon + self.verbalize_field("minutes", before))
        clock_verb += seconds_verb.ques
        loose_verb = prep_verb + hours_verb + (colon + self.verbalize_field("minutes", bare_minutes) | insert(":00"))
        noon_verb = prep_verb.ques + self.verbalize_field("hours", special) + insert(":00")
        self.verbalizer = self.delete_tokens(prep_verb.ques + clock_verb | loose_verb | noon_verb)
