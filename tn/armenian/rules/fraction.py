# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from pynini.lib.pynutil import delete, insert

from tn.processor import Processor


class Fraction(Processor):
    """Fractions, e.g. ``4/5`` -> ``չորս հինգերորդ``.

    The numerator is read as a cardinal and the denominator as an ordinal.
    """

    def __init__(self, cardinal, ordinal):
        super().__init__("fraction", token_orders={})
        numerator = cardinal.graph
        denominator = ordinal.denominator
        tagger = self.tag_field("numerator", numerator) + delete("/") + insert(" ")
        tagger += self.tag_field("denominator", denominator)
        self.tagger = self.add_tokens(tagger)
        verbalizer = self.verbalize_field("numerator", numerator) + insert(" ")
        verbalizer += self.verbalize_field("denominator", denominator, leading_space=True)
        self.verbalizer = self.delete_tokens(verbalizer)
