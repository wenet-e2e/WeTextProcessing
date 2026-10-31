# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
from pynini.lib import pynutil

from itn.russian.rules.utils import RU_ALPHA, data_path, latin_to_cyrillic
from itn.utils import load_labels
from tn.processor import Processor


class Electronic(Processor):
    """Spoken e-mail addresses, e.g. "эй би собака эн ди точка ру" -> "ab@nd.ru"."""

    def __init__(self):
        super().__init__(name="electronic", ordertype="itn")
        symbol_chars = pynini.union(*[row[0] for row in load_labels(data_path("electronic/symbols.tsv"))])
        username = self.ALPHA + pynini.closure(self.ALPHA | self.DIGIT | symbol_chars)
        domain = self.ALPHA + pynini.closure(self.ALPHA | self.DIGIT | "-" | ".") + (self.ALPHA | self.DIGIT)

        # Written -> spoken spellings as in TN, inverted below.
        digit = pynini.string_file(data_path("numbers/digits_nominative_case.tsv")).optimize()
        symbols = pynini.string_file(data_path("electronic/symbols.tsv")).optimize()
        char = pynini.difference(self.VCHAR, pynini.union('"', " "))
        space = pynutil.insert(" ")
        spell = pynini.closure(latin_to_cyrillic() | RU_ALPHA | " ").optimize()

        unit = (
            pynutil.add_weight(digit, 1.09)
            | pynutil.add_weight(symbols + pynini.closure(space + symbols), 1.09)
            | pynutil.add_weight(char, 1.1)
        )
        user = unit + pynini.closure(space + unit)
        self.username_graph = (username @ user @ spell).invert().optimize()

        alnum = digit | self.ALPHA
        server_default = (alnum + space).plus + pynini.closure(symbols + space) + (alnum + space).plus
        server_common = pynini.string_file(data_path("electronic/server_name.tsv")) + space
        dot = pynini.cross(".", "точка ")
        domain_default = pynini.closure(char + space) + dot + char + pynini.closure(space + char)
        domain_common = dot + pynini.string_file(data_path("electronic/domain.tsv"))
        domain_reading = (
            (pynutil.add_weight(server_common, 1.09) | pynutil.add_weight(server_default, 1.1))
            + (pynutil.add_weight(domain_common, 1.09) | pynutil.add_weight(domain_default, 1.1))
        )
        self.domain_graph = (domain @ domain_reading @ spell).invert().optimize()

        self.tagger = self.add_tokens(
            self.tag_field("username", self.username_graph) + pynutil.delete(" собака ") + pynutil.insert(" ")
            + self.tag_field("domain", self.domain_graph)
        )
        self.verbalizer = self.delete_tokens(
            self.verbalize_field("username", self.username_graph) + pynutil.delete(" ") + pynutil.insert("@")
            + self.verbalize_field("domain", self.domain_graph)
        )
