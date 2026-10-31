# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from itn.utils import get_abs_path
from tn.processor import Processor
from tn.utils import load_labels


class Measure(Processor):
    """Quantities with units, e.g. ``عشرون في المائة`` -> ``20%``.

    The quantity is a cardinal, decimal or fraction (flat raw fields reusing
    those rules' graphs) followed by a spoken unit; the unit symbol is
    attached without a space.
    """

    def __init__(self, cardinal, decimal, fraction):
        super().__init__(name="measure", ordertype="itn")
        names = {}
        for symbol, name in load_labels(get_abs_path("arabic/data/measure/measurements.tsv")):
            names.setdefault(name, symbol)
        self.units = pynini.string_map([(pynini.escape(name), pynini.escape(symbol))
                                        for name, symbol in names.items()]).optimize()
        negative = cardinal.negative
        sign = self.tag_field("negative", negative) + self.DELETE_EXTRA_SPACE
        quantity = self.tag_field("integer", cardinal.graph) | decimal.tag_number() | fraction.tag_number()
        units = self.DELETE_EXTRA_SPACE + self.tag_field("units", self.units)
        self.tagger = self.add_tokens(sign.ques + quantity + units)

        sign = self.verbalize_field("negative", negative) + delete(" ")
        quantity = self.verbalize_field("integer", cardinal.graph) | decimal.verbalize_number()
        quantity |= fraction.verbalize_number()
        units = self.verbalize_field("units", self.units, True)
        self.verbalizer = self.delete_tokens(sign.ques + quantity + units)
