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

from tn.processor import Processor


class Prefixed(Processor):
    """Re-exposes an existing rule under a language-prefixed token name.

    Spanish and English rules share token names ("cardinal", "money", ...)
    but not field layouts or verbalizers, so the English rules are renamed
    ("en_cardinal") to keep each raw tag bound to its own verbalizer and
    field order. Only the leading token name is rewritten; the raw fields
    produced by the wrapped tagger are untouched.
    """

    def __init__(self, rule, prefix="en_"):
        super().__init__(prefix + rule.name, ordertype="itn")
        rename = pynini.cross(rule.name + " {", self.name + " {") + self.VSIGMA
        self.tagger = (rule.tagger @ rename).optimize()
        self.verbalizer = (pynini.invert(rename) @ rule.verbalizer).optimize()
