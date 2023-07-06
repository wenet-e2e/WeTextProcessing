# Copyright (c) 2022 Zhendong Peng (pzd17@tsinghua.org.cn)
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

from tn.processor import Processor

from pynini import difference, string_file
from pynini.lib.pynutil import delete
from pynini.lib.tagger import Tagger


class PostProcessor(Processor):

    def __init__(self, remove_puncts=False, tag_oov=False):
        super().__init__(name='postprocessor')
        puncts = string_file('tn/chinese/data/char/punctuations_zh.tsv')
        zh_charset_std = string_file(
            'tn/chinese/data/char/charset_national_standard_2013_8105.tsv')
        zh_charset_ext = string_file(
            'tn/chinese/data/char/charset_extension.tsv')

        processor = self.build_rule('')
        if remove_puncts:
            processor @= self.build_rule(delete(puncts | self.PUNCT))

        if tag_oov:
            charset = (zh_charset_std | zh_charset_ext | puncts | self.DIGIT
                       | self.ALPHA | self.PUNCT | self.SPACE)
            oov = difference(self.VCHAR, charset)
            processor @= Tagger('oov', oov, self.VSIGMA)._tagger

        self.processor = processor.optimize()
