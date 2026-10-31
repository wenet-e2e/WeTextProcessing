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
from pynini.lib.pynutil import delete

from itn.hungarian.rules.utils import data_labels, naive_inflector
from tn.processor import Processor

TOKEN_ORDER = {}

# Clock units are written by the time rule.
SKIPPED_UNITS = {"ó", "mp"}


def unit_rows():
    """Unit rows with lower-case abbreviations (``l`` before ``L``) first, otherwise in file order."""

    rows = data_labels("measures/measurements.tsv")
    return sorted(rows, key=lambda row: row[0] != row[0].lower())


def units():
    """Spoken units (with case endings) -> abbreviations; the first abbreviation wins."""

    written = {}
    for row in unit_rows():
        key, word = row[0], row[1]
        if key in SKIPPED_UNITS:
            continue
        written.setdefault(word, key)
        if " " not in word:
            for abbr, spoken in naive_inflector(key.lower(), word):
                written.setdefault(spoken, abbr)
    return pynini.string_map([(spoken, abbr) for spoken, abbr in written.items()]).optimize()


class Measure(Processor):
    """Measures, e.g. ``három milliméterben`` -> ``3mm-ben``, ``mínusz tizenkettő kilogramm`` -> ``-12kg``.

    The number fields and the spoken (possibly inflected) unit stay raw; the
    unit abbreviation is attached to the number.
    """

    def __init__(self, cardinal, decimal, fraction):
        super().__init__("measure", ordertype="itn", token_orders=TOKEN_ORDER)
        unit = units()
        bare = {}
        for row in unit_rows():
            if " " not in row[1]:
                bare.setdefault(row[1], row[0])
        bare = pynini.string_map(list(bare.items()))
        per = pynini.cross(" per ", "/") + bare
        self.units = (unit + per.ques).optimize()

        number_t = self.tag_field("integer", cardinal.graph) | decimal.unsigned_tagger | fraction.unsigned_tagger
        units_t = pynini.accep(" ") + self.tag_field("units", self.units)
        self.tagger = self.add_tokens(cardinal.sign_tagger + number_t + units_t)

        number_v = (self.verbalize_field("integer", cardinal.graph)
                    | decimal.unsigned_verbalizer | fraction.unsigned_verbalizer)
        units_v = delete(" ") + self.verbalize_field("units", self.units)
        self.verbalizer = self.delete_tokens(cardinal.sign_verbalizer + number_v + units_v)
