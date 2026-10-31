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
"""Hebrew number words.

Counting uses the feminine forms (``שלוש``, ``שלוש עשרה``). ``ו`` joins the
last word of a number when that word is a unit, a teen or ten, or round tens
that directly follow a thousand/million word: ``מאה ושש עשרה``, ``אלף וארבע``,
``עשרת אלפים ועשרים``, but ``מאתיים שלושים`` and ``אלפיים מאה``. The thousand
and million words are masculine, so a final ``אחד`` counts them (``מאה ואחד
אלף``) and a single-digit million count is masculine (``שלושה מיליון``).
"""

import pynini
from pynini.lib.pynutil import delete

from tn.hebrew.rules.utils import DIGIT, NONZERO

FEMININE = ["אפס", "אחת", "שתיים", "שלוש", "ארבע", "חמש", "שש", "שבע", "שמונה", "תשע"]
MASCULINE = ["אפס", "אחד", "שניים", "שלושה", "ארבעה", "חמישה", "שישה", "שבעה", "שמונה", "תשעה"]
TENS = ["", "", "עשרים", "שלושים", "ארבעים", "חמישים", "שישים", "שבעים", "שמונים", "תשעים"]
HUNDREDS = ["", "מאה", "מאתיים"] + [FEMININE[digit] + " מאות" for digit in range(3, 10)]
THOUSANDS = {1: "אלף", 2: "אלפיים", 3: "שלושת אלפים", 4: "ארבעת אלפים", 5: "חמשת אלפים", 6: "ששת אלפים",
             7: "שבעת אלפים", 8: "שמונת אלפים", 9: "תשעת אלפים", 10: "עשרת אלפים"}
ORDINALS = ["", "ראשון", "שני", "שלישי", "רביעי", "חמישי", "שישי", "שביעי", "שמיני", "תשיעי", "עשירי"]


def _below_hundred(number, masculine=False):
    units = MASCULINE if masculine else FEMININE
    if number < 10:
        return [("unit", units[number])] if number else []
    if number == 10:
        return [("unit", "עשרה" if masculine else "עשר")]
    if number < 20:
        return [("unit", units[number - 10] + (" עשר" if masculine else " עשרה"))]
    tens, unit = divmod(number, 10)
    return [("tens", TENS[tens])] + ([("unit", units[unit])] if unit else [])


def group(number, higher=False, masculine=False, counts_masculine=False):
    """Reads 0-999; ``higher`` tells that a thousand/million word precedes."""

    hundreds, rest = divmod(number, 100)
    pieces = [("hundreds", HUNDREDS[hundreds])] if hundreds else []
    pieces += _below_hundred(rest, masculine)
    if counts_masculine and rest % 10 == 1 and rest != 11:
        pieces[-1] = ("unit", "אחד")
    if pieces and (len(pieces) > 1 or higher):
        kind, word = pieces[-1]
        if kind == "unit" or kind == "tens" and len(pieces) == 1:
            pieces[-1] = (kind, "ו" + word)
    return " ".join(word for _, word in pieces)


def thousands(number):
    return THOUSANDS.get(number) or group(number, counts_masculine=True) + " אלף"


def millions(number):
    if number == 1:
        return "מיליון"
    if number == 2:
        return "שני מיליון"
    if number < 10:
        return MASCULINE[number] + " מיליון"
    return group(number, counts_masculine=True) + " מיליון"


def cardinal(number):
    """Reads 0-999,999,999."""

    if number == 0:
        return "אפס"
    high, rest = divmod(number, 1000000)
    middle, low = divmod(rest, 1000)
    words = [millions(high)] if high else []
    if middle:
        words.append(thousands(middle))
    if low:
        words.append(group(low, higher=bool(words)))
    return " ".join(words)


def _map(pairs):
    return pynini.string_map(pairs).optimize()


def _padded(reader):
    """Maps a zero-padded group to `` <words>``, or ``000`` to nothing."""

    return _map([("%03d" % number, " " + reader(number) if number else "") for number in range(1000)])


def cardinal_graph():
    """Maps digits without leading zeros (and ``0``) to the cardinal reading."""

    small = _map([(str(number), cardinal(number)) for number in range(1000)])
    tail = _padded(lambda number: group(number, higher=True))
    many_thousands = _map([(str(number), thousands(number)) for number in range(1, 1000)])
    many_millions = _map([(str(number), millions(number)) for number in range(1, 1000)])
    graph = small | many_thousands + tail | many_millions + _padded(thousands) + tail
    return graph.optimize()


def grouped_digits():
    """Accepts ``1,000`` or ``1000`` style integers and deletes the commas."""

    plain = NONZERO + DIGIT.star | "0"
    grouped = NONZERO + pynini.closure(DIGIT, 0, 2) + (delete(",") + DIGIT**3).plus
    return (plain | grouped).optimize()


def masculine_graph(limit):
    return _map([(str(number), group(number, masculine=True)) for number in range(1, limit + 1)])


def ordinal_graph():
    return _map([(str(number), ORDINALS[number]) for number in range(1, 11)])


def digit_by_digit():
    """``0453`` -> ``אפס ארבע חמש שלוש``; a lone one is ``אחד``."""

    digit = _map([(str(number), "אחד" if number == 1 else FEMININE[number]) for number in range(10)])
    return (digit + (pynini.cross("", " ") + digit).star).optimize()
