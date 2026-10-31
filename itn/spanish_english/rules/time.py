# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from itn.spanish.rules.time import Time as SpanishTime


class Time(SpanishTime):
    """Spanish clock times written the code-mixed way.

    The code-mixed grammar writes an article-less time ("dos y media de la
    tarde") with the English two-digit hour ("02:30 p.m."); times with an
    article keep the Spanish form ("las 2:00 p.m."). The raw tag is the
    Spanish one; only the verbalizer output is padded.
    """

    def __init__(self, cased_input=False):
        super().__init__(cased_input)
        digit = self.DIGIT
        pad = pynini.union(
            pynutil.insert("0") + digit + ":",
            digit + digit + ":",
            pynini.difference(self.VCHAR, digit),
        ) + self.VSIGMA
        self.verbalizer = (self.verbalizer @ pad.optimize()).optimize()
