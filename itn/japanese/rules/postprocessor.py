# Copyright (c) 2024 Logan Liu (2319277867@qq.com)
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
from tn.utils import get_abs_path

from pynini import string_file, difference
from pynini.lib.pynutil import delete
from pynini.lib.tagger import Tagger


class PostProcessor(Processor):

    def __init__(self,
                 remove_interjections=False,
                 remove_puncts=False,
                 tag_oov=False):
        super().__init__(name='postprocessor')
        blacklist = string_file(
            get_abs_path('../itn/japanese/data/default/blacklist.tsv'))
        puncts = string_file(
            get_abs_path('../itn/japanese/data/char/punctuations_ja.tsv'))
        # 片假名/平假名/浊音/半浊音/小写假名
        ja_charset_std = string_file(
            get_abs_path(
                '../itn/japanese/data/char/hiragana_and_katakana.tsv'))
        # 日语常用汉字表
        ja_charset_ext = string_file(
            get_abs_path('../itn/japanese/data/char/common_chinese_char.tsv'))

        processor = self.build_rule('')
        if remove_interjections:
            processor @= self.build_rule(delete(blacklist))

        if remove_puncts:
            processor @= self.build_rule(delete(puncts | self.PUNCT))

        if tag_oov:
            charset = (ja_charset_std
                       | ja_charset_ext | puncts | self.DIGIT
                       | self.ALPHA | self.PUNCT | self.SPACE)
            oov = difference(self.VCHAR, charset)
            processor @= Tagger('oov', oov, self.VSIGMA)._tagger

        self.processor = processor.optimize()
