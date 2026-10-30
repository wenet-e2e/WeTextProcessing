# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pynini
from pynini.lib import pynutil

from tn.processor import Processor
from tn.utils import get_abs_path, load_labels


TOKEN_ORDER = {}


class WhiteList(Processor):
    def __init__(self, deterministic=True):
        super().__init__("whitelist", token_orders=TOKEN_ORDER)
        self.deterministic = deterministic
        labels = load_labels(get_abs_path("german/data/whitelist.tsv"))
        graph = pynini.string_map(labels)
        if not deterministic:
            lower = pynini.string_map([[row[0].lower()] + row[1:] for row in labels])
            graph = graph | pynutil.add_weight(lower, 0.0001)
            graph = graph | pynini.string_file(get_abs_path("german/data/measure/measurements.tsv"))
        self.graph = (graph @ pynini.cdrewrite(pynini.cross("\u00a0", " "), "", "", self.VSIGMA)).optimize()
        self.tagger = self.add_tokens(self.tag_field("value", self.graph))
        self.verbalizer = self.delete_tokens(self.verbalize_field("value", self.graph))
