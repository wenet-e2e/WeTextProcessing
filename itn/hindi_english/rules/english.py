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
"""Code-mixed glue around the English ITN rules.

The English rules are reused unchanged except for three code-mixed details
(Indian lakh/crore numbering, a comma before the year of a full date as in the
Hindi date verbalizer, and plural "degrees" temperature units), plus optional
cased input. Cased input is handled field by field: a field also accepts its
spoken form with a capitalized first letter (the raw text, e.g. ``July``, is
kept in the tag), and the verbalizer lower-cases that letter before converting
and re-capitalizes a written form that starts with a lower-case letter
(``July`` -> ``July``, ``Twenty`` -> ``20``).
"""

import string

import pynini
from pynini import closure, union
from pynini.lib import utf8
from pynini.lib.pynutil import delete, insert

from itn.english.rules.cardinal import Cardinal as EnCardinal
from itn.english.rules.date import Date as EnDate
from itn.english.rules.decimal import Decimal as EnDecimal
from itn.english.rules.electronic import Electronic as EnElectronic
from itn.english.rules.measure import Measure as EnMeasure
from itn.english.rules.money import Money as EnMoney
from itn.english.rules.ordinal import Ordinal as EnOrdinal
from itn.english.rules.telephone import Telephone as EnTelephone
from itn.english.rules.time import Time as EnTime
from itn.english.rules.whitelist import Whitelist as EnWhitelist
from itn.utils import get_abs_path

_SIGMA = utf8.VALID_UTF8_CHAR.star
_LOWER = union(*string.ascii_lowercase)
_TO_LOWER = union(*[pynini.cross(u, lo) for u, lo in zip(string.ascii_uppercase, string.ascii_lowercase)])
# "July" -> "july"; only strings with a capitalized first letter.
LOWER_FIRST = (_TO_LOWER + _SIGMA).optimize()
# Written form re-capitalized only when it starts with a lower-case letter ("july" -> "July", "20" -> "20").
RECAPITALIZE = (pynini.invert(_TO_LOWER) + _SIGMA | pynini.difference(utf8.VALID_UTF8_CHAR, _LOWER) + _SIGMA
                | "").optimize()

# Indian numbering magnitudes, highest first (each takes a 1-99 multiplier).
INDIAN_MAGNITUDES = ("shankh", "padma", "nil", "kharab", "arab", "crore", "lakh")


def data_file(rel_path):
    """Loads hi_en ``written<TAB>spoken`` data as spoken -> written."""

    return pynini.string_file(get_abs_path("hindi_english/data/" + rel_path)).invert().optimize()


class CasedFields:
    """Mixin: with ``cased=True`` every tagged field also accepts a capitalized first letter."""

    def __init__(self, *args, cased=False, **kwargs):
        self.cased = cased
        super().__init__(*args, **kwargs)

    def tag_field(self, name, graph):
        if self.cased:
            graph = graph | (LOWER_FIRST @ graph)
        return super().tag_field(name, graph)

    def verbalize_field(self, name, graph=None, leading_space=False):
        if self.cased and graph is not None:
            graph = graph | (LOWER_FIRST @ graph @ RECAPITALIZE)
        return super().verbalize_field(name, graph, leading_space)


class Cardinal(CasedFields, EnCardinal):
    """English cardinal plus Indian numbering: ``twenty one crore ninety eight lakh ...`` -> ``219836593``."""

    def build_tagger(self):
        super().build_tagger()
        indian = self.indian_graph()
        self.graph = (self.graph | indian).optimize()
        self.graph_no_exception = self.graph
        self.integer = (self.integer | indian).optimize()
        optional_minus = closure(self.tag_field("negative", self.negative) + delete(" ") + insert(" "), 0, 1)
        self.tagger = self.add_tokens(optional_minus + self.tag_field("integer", self.integer))

    def indian_graph(self):
        digit = pynini.string_file(get_abs_path("english/data/numbers/digit.tsv"))
        two_digit = self.graph_two_digit
        two_padded = two_digit | insert("0") + digit
        ds = delete(" ")
        hundreds = digit + ds + delete("hundred") + (ds + two_digit | ds + insert("0") + digit | insert("00"))
        up_to_999_padded = hundreds | insert("0") + two_digit | insert("00") + digit

        # Parsed over the input with one leading space, so every present component is " <number> <magnitude>".
        graph = pynini.accep("")
        for magnitude in INDIAN_MAGNITUDES + ("thousand", ):
            words = pynini.union(magnitude, magnitude + "s") if magnitude != "thousand" else pynini.accep(magnitude)
            graph += ds + two_padded + ds + delete(words) | insert("00")
        graph += ds + up_to_999_padded | insert("000")
        graph = (insert(" ") + _SIGMA) @ graph
        has_magnitude = _SIGMA + pynini.union(*INDIAN_MAGNITUDES) + _SIGMA
        graph = has_magnitude @ graph
        graph = graph @ (delete(closure("0")) + pynini.difference(self.DIGIT, "0") + closure(self.DIGIT))
        delete_and = self.build_rule(delete("and "), " ", self.ALPHA)
        return (delete_and @ graph).optimize()


class Ordinal(CasedFields, EnOrdinal):
    pass


class Decimal(CasedFields, EnDecimal):
    pass


class Date(CasedFields, EnDate):
    """English date written like the Hindi date: a comma separates day and year (``july 25, 2012``)."""

    def build_verbalizer(self):
        month = self.verbalize_field("month", self.month)
        day = self.verbalize_field("day", self.day | self.day_dmy)
        period = self.verbalize_field("day", self.day_quarter | self.day_half)
        year = self.verbalize_field(
            "year", self.year | self.year_decade | self.year_bc | self.year_century | self.year_millennium)
        delete_po = delete("preserve_order:") + self.DELETE_SPACE + delete('"true"')
        space = self.DELETE_SPACE + insert(" ")
        comma_year = closure(self.DELETE_SPACE + insert(", ") + year, 0, 1)

        graph = pynini.union(
            month + space + day + comma_year,
            month + closure(space + year, 0, 1),
            day + space + month + comma_year,
            period + space + year,
            year,
        )
        self.verbalizer = self.delete_tokens(graph + self.DELETE_SPACE + delete_po)


class Measure(CasedFields, EnMeasure):
    """English measure that also accepts plural temperature units (``five degrees celsius`` -> ``5 °C``)."""

    def build_tagger(self):
        super().build_tagger()
        self.units = (self.units | data_file("measurements.tsv")).optimize()
        value = self.tag_field("value", self.value)
        units = self.tag_field("units", self.units)
        self.tagger = self.add_tokens(value + delete(" ") + insert(" ") + units)


class Money(CasedFields, EnMoney):
    pass


class Time(CasedFields, EnTime):
    pass


class Telephone(CasedFields, EnTelephone):
    pass


class Electronic(CasedFields, EnElectronic):
    pass


class Whitelist(CasedFields, EnWhitelist):
    pass
