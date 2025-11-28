# Copyright (c) 2022 Zhendong Peng (pzd17@tsinghua.org.cn)
# Copyright (c) 2023 Xingchen Song (sxc19@mails.tsinghua.edu.cn)
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

from pynini import string_file
from pynini.lib.pynutil import delete

from tn.processor import Processor
from tn.utils import get_abs_path


class PostProcessor(Processor):

    def __init__(self, remove_interjections=True):
        super().__init__(name="postprocessor")
        blacklist = string_file(get_abs_path("../itn/chinese/data/default/blacklist.tsv"))

        processor = self.VSIGMA
        if remove_interjections:
            processor @= self.build_rule(delete(blacklist))
        self.processor = processor
