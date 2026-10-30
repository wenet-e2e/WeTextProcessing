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
from tn.utils import get_abs_path


TOKEN_ORDER = {}


class Ordinal(Processor):
    """Preserve written ordinal digits and endings; generate all five inflections."""

    def __init__(self, cardinal, deterministic=True):
        super().__init__("ordinal", token_orders=TOKEN_ORDER)
        self.deterministic = deterministic
        self.stem = pynini.union(*[
            pynini.string_file(get_abs_path("german/data/ordinals/" + name + ".tsv")).invert()
            for name in ("digit", "ties", "thousands")
        ]).optimize()
        endings = pynini.union("ten", "tem", "ter", "tes", "te")
        self.suffix = pynini.cdrewrite(
            pynini.closure(self.stem, 0, 1) + pynutil.insert(endings, weight=0.01),
            "", "[EOS]", self.VSIGMA,
        ).optimize()
        self.graph = (cardinal.graph @ self.suffix).optimize()
        written_suffix = pynutil.delete(".") | pynutil.add_weight(pynutil.delete(endings), 0.0001)
        self.tagger = self.add_tokens(
            self.tag_field("integer", self.graph) + pynutil.insert(" ")
            + self.tag_field("suffix", written_suffix)
        )
        self.verbalizer = self.delete_tokens(
            self.verbalize_field("integer", self.graph)
            + self.verbalize_field("suffix", written_suffix, leading_space=True)
        )
