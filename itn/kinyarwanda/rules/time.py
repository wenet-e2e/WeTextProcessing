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

from itn.kinyarwanda.rules.utils import string_map
from tn.processor import Processor


class Time(Processor):
    """Rwandan clock times, e.g. ``saa cyenda n'iminota mirongo itatu`` -> ``3:30``.

    The tagger drops ``saa`` and the ``n'`` connector and keeps the hour
    words and the minute phrase (``umunota umwe``, ``iminota ...``). A time
    without minutes is written on the full hour.
    """

    def __init__(self):
        super().__init__(name="time", ordertype="itn", token_orders={})
        hours = string_map("kinyarwanda/data/time/hours.tsv")
        minutes = string_map("kinyarwanda/data/time/minutes.tsv")
        tagger = delete("saa ") + self.tag_field("hours", hours)
        tagger += (delete(" n'") + insert(" ") + self.tag_field("minutes", minutes)).ques
        self.tagger = self.add_tokens(tagger)
        verbalizer = self.verbalize_field("hours", hours) + insert(":")
        verbalizer += self.verbalize_field("minutes", minutes, leading_space=True) | insert("00")
        self.verbalizer = self.delete_tokens(verbalizer)
