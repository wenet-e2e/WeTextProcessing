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

from itn.italian.rules.utils import load_rows, string_map
from tn.processor import Processor


class Measure(Processor):
    """Quantities with units, e.g. ``cinque chilometri per secondo`` -> ``5 km/s``.

    The tagger keeps the spoken number fields and unit words; the
    verbalizer writes the number and the unit symbol (``%`` is attached).
    Singular and plural unit names are accepted after any number. Time units
    (``giorni``, ``ore``, ``minuti``, ``secondi``) are only converted as
    denominators, since ``sette giorni`` normally stays in words.
    """

    def __init__(self, cardinal, decimal):
        super().__init__("measure", token_orders={})
        rows = load_rows("italian/data/measure/units.tsv")
        units = string_map([(row[0], spoken) for row in rows for spoken in row[1:]])
        per_units = [(row[0], row[1]) for row in rows]
        per_units += [(row[0], row[1]) for row in load_rows("italian/data/measure/denominators.tsv")]
        denominator = pynini.cross(" per ", "/") + string_map(per_units)
        percent = (units @ "%").optimize()
        others = (units @ (self.VSIGMA - "%")).optimize()
        unit_graph = (percent | others + denominator.ques).optimize()

        sign_tag = (self.tag_field("negative", cardinal.sign) + pynini.accep(" ")).ques
        number_tag = self.tag_field("integer", cardinal.counted) | decimal.fraction_tagger
        units_tag = pynini.accep(" ") + self.tag_field("units", unit_graph)
        self.tagger = self.add_tokens(sign_tag + number_tag + units_tag)

        sign_verb = (self.verbalize_field("negative", cardinal.sign) + delete(" ")).ques
        number_verb = self.verbalize_field("integer", cardinal.counted) | decimal.fraction_verbalizer
        units_verb = delete(" ") + insert(" ") + self.verbalize_field("units", others + denominator.ques)
        percent_verb = delete(" ") + self.verbalize_field("units", percent)
        self.verbalizer = self.delete_tokens(sign_verb + number_verb + (units_verb | percent_verb))
