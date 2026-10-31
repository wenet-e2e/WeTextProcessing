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

The English rules are reused unchanged except for two code-mixed details
(Indian lakh/crore numbering in the cardinal, and a few es_en measurement
units such as ``gigabits per second`` -> ``gbps``), plus optional cased input.
Cased input is handled field by field: a field also accepts its spoken form
with a capitalized first letter (the raw text, e.g. ``Billion``, is kept in
the tag), and the verbalizer lower-cases that letter before converting and
re-capitalizes a written form that starts with a lower-case letter
(``Billion`` -> ``Billion``, ``Twenty`` -> ``20``).
"""

import string

import pynini
from pynini import closure, union
from pynini.lib import utf8
from pynini.lib.pynutil import add_weight, delete, insert

from itn.english.rules.cardinal import Cardinal as EnCardinal
from itn.english.rules.date import Date as EnDate
from itn.english.rules.decimal import Decimal as EnDecimal
from itn.english.rules.electronic import Electronic as EnElectronic
from itn.english.rules.measure import Measure as EnMeasure
from itn.english.rules.money import Money as EnMoney
from itn.english.rules.ordinal import Ordinal as EnOrdinal
from itn.english.rules.telephone import Telephone as EnTelephone
from itn.english.rules.time import Time as EnTime
from itn.utils import get_abs_path

_SIGMA = utf8.VALID_UTF8_CHAR.star
_LOWER = union(*string.ascii_lowercase)
_TO_LOWER = union(*[pynini.cross(u, lo) for u, lo in zip(string.ascii_uppercase, string.ascii_lowercase)])
# "Billion" -> "billion"; only strings with a capitalized first letter.
LOWER_FIRST = (_TO_LOWER + _SIGMA).optimize()
# Written form re-capitalized only when it starts with a lower-case letter ("billion" -> "Billion", "20" -> "20").
RECAPITALIZE = (pynini.invert(_TO_LOWER) + _SIGMA | pynini.difference(utf8.VALID_UTF8_CHAR, _LOWER) + _SIGMA
                | "").optimize()
# "GigaBits" -> "gigabits": every letter lower-cased (used for unit names).
LOWER_ALL = (_TO_LOWER | utf8.VALID_UTF8_CHAR).star.optimize()

# Indian numbering magnitudes, highest first (each takes a 1-99 multiplier).
INDIAN_MAGNITUDES = ("shankh", "padma", "nil", "kharab", "arab", "crore", "lakh")


def data_file(rel_path):
    """Loads es_en ``written<TAB>spoken`` data as spoken -> written."""

    return pynini.string_file(get_abs_path("spanish_english/data/" + rel_path)).invert().optimize()


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
    """English cardinal plus Indian numbering: ``twenty one crore ninety eight lakh ...`` -> ``219836593``.

    The Indian graph is also merged into ``graph``, so decimal, money and
    measure values accept it as well.
    """

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
    pass


class Measure(CasedFields, EnMeasure):
    """English measure with the es_en units (``nine gigabits per second`` -> ``9 gbps``).

    With ``cased=True`` unit names are matched case-insensitively
    (``GigaBits per second``) and written as the plain unit symbol.
    """

    def build_tagger(self):
        super().build_tagger()
        # Unweighted, so the es_en units win over the English plural + weighted "per" unit ("gb/s").
        self.units = (self.units | data_file("measurements.tsv")).optimize()
        if self.cased:
            self.units = (self.units | LOWER_ALL @ self.units).optimize()
        value = self.tag_field("value", self.value)
        units = super(CasedFields, self).tag_field("units", self.units)
        self.tagger = self.add_tokens(value + delete(" ") + insert(" ") + units)

    def build_verbalizer(self):
        value = self.verbalize_field("value", self.value)
        units = super(CasedFields, self).verbalize_field("units", self.units)
        self.verbalizer = self.delete_tokens(value + self.DELETE_SPACE + insert(" ") + units)


class Money(CasedFields, EnMoney):
    pass


class Time(CasedFields, EnTime):
    """English time; with ``cased=True`` also upper-case suffixes and zones (``A M E S T`` -> ``A.M. EST``)."""

    def tag_field(self, name, graph):
        return super().tag_field(name, self.cased_variants(name, graph))

    def verbalize_field(self, name, graph=None, leading_space=False):
        return super().verbalize_field(name, self.cased_variants(name, graph), leading_space)

    def cased_variants(self, name, graph):
        if not self.cased or name not in ("noon", "zone"):
            return graph
        if name == "noon":
            cased = pynini.string_file(get_abs_path("english/data/time/time_suffix_cased.tsv"))
        else:
            cased = pynini.string_file(get_abs_path("english/data/time/time_zone_cased.tsv")).invert()
        return (graph | cased).optimize()


class Telephone(CasedFields, EnTelephone):
    pass


class Electronic(EnElectronic):
    """English electronic address with the server names (``n vidia`` -> ``nvidia``).

    The English rule already keeps upper-case letters (``A B C`` -> ``ABC``),
    so ``cased=True`` only adds a lower-cased server name (``N vidia`` ->
    ``nvidia``), an upper-case ``W W W`` and a literal ``.`` after it.
    """

    def __init__(self, cased=False):
        self.cased = cased
        super().__init__()

    def build_tagger(self):
        ds = delete(" ")
        digit = pynini.string_file(get_abs_path("english/data/numbers/digit.tsv"))
        zero = pynini.string_file(get_abs_path("english/data/numbers/zero.tsv"))
        symbols = pynini.string_file(get_abs_path("english/data/electronic/symbols.tsv")).invert()
        server = pynini.string_file(get_abs_path("english/data/electronic/server_name.tsv"))
        if self.cased:
            server |= LOWER_FIRST @ server

        char = self.ALPHA | digit | zero
        word = add_weight(closure(self.ALPHA, 2), 0.1)
        token = char | symbols | word
        first_token = char | pynini.difference(word, pynini.accep("dot")) | add_weight(server, -0.1)
        component = first_token + closure(ds + token)

        dot = pynini.cross("dot", ".")
        domain = component + (ds + dot + ds + component).plus

        http = pynini.cross("h t t p", "http")
        https = pynini.cross("h t t p s", "https")
        protocol = (http | https) + pynini.cross(" colon slash slash ", "://")
        www = pynini.cross("w w w", "www")
        www_dot = ds + dot + ds
        if self.cased:
            www |= pynini.cross("W W W", "www")
            www_dot |= closure(ds, 0, 1) + "." + closure(ds, 0, 1)

        self.username = component
        self.domain = delete("at") + ds + domain
        self.protocol = protocol + closure(www + www_dot, 0, 1) + domain | www + www_dot + domain | domain

        username = self.tag_field("username", self.username)
        domain_field = self.tag_field("domain", self.domain)
        graph_email = add_weight(username + ds + insert(" ") + domain_field, -0.001)
        graph_url = self.tag_field("protocol", self.protocol)
        self.tagger = self.add_tokens(graph_email | graph_url)
