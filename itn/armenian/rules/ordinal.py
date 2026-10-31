# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
from pynini.lib.pynutil import add_weight, insert

from itn.armenian.rules.utils import preferred_map
from tn.processor import Processor


class Ordinal(Processor):
    """Spoken ordinals, e.g. ``հիսունյոթերորդ`` -> ``57-րդ``, ``առաջին`` -> ``1-ին``.

    The tagger keeps the spoken ordinal; the verbalizer strips the ordinal
    ending, reads the cardinal and appends the written suffix.
    """

    def __init__(self, cardinal):
        super().__init__(name="ordinal", ordertype="itn", token_orders={})
        digit = preferred_map("armenian/data/ordinals/digit.tsv")
        ending = digit | pynini.cross("երորդ", "")
        self.graph = ((self.VSIGMA + ending) @ cardinal.graph).optimize()
        suffix = pynini.accep("1") + insert("-ին") | add_weight(self.DIGIT.plus + insert("-րդ"), 0.01)
        self.tagger = self.add_tokens(self.tag_field("integer", self.graph))
        self.verbalizer = self.delete_tokens(self.verbalize_field("integer", self.graph @ suffix))
