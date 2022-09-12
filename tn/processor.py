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

from tn.token_parser import TokenParser

from pynini import cdrewrite, cross, difference, escape, shortestpath, union
from pynini.export import export
from pynini.lib import byte, utf8
from pynini.lib.pynutil import delete, insert


class Processor:

    def __init__(self, name, ordertype="tn"):
        self.ALPHA = byte.ALPHA
        self.DIGIT = byte.DIGIT
        self.PUNCT = byte.PUNCT
        self.SPACE = byte.SPACE | u'\u00A0'
        self.VCHAR = utf8.VALID_UTF8_CHAR
        self.VSIGMA = self.VCHAR.star

        CHAR = difference(self.VCHAR, union('\\', '"'))
        self.CHAR = (CHAR | cross('\\', '\\\\\\') | cross('"', '\\"'))
        self.SIGMA = (CHAR | cross('\\\\\\', '\\') | cross('\\"', '"')).star

        self.name = name
        self.parser = TokenParser(ordertype)
        self.tagger = None
        self.verbalizer = None

    def build_rule(self, fst, l='', r=''):
        rule = cdrewrite(fst, l, r, self.VSIGMA)
        return rule

    def add_tokens(self, tagger):
        tagger = insert(f"{self.name} {{ ") + tagger + insert(' }')
        return tagger.optimize()

    def delete_tokens(self, verbalizer):
        verbalizer = (delete(f"{self.name}") + delete(' { ') + verbalizer +
                      delete(' }'))
        return verbalizer.optimize()

    def build_verbalizer(self):
        verbalizer = delete('value: "') + self.SIGMA + delete('"')
        self.verbalizer = self.delete_tokens(verbalizer)

    def export(self, file_name):
        exporter = export.Exporter(file_name)
        exporter['tagger'] = self.tagger.optimize()
        exporter['verbalizer'] = self.verbalizer.optimize()
        exporter.close()

    def tag(self, input):
        lattice = input @ self.tagger
        return shortestpath(lattice, nshortest=1, unique=True).string()

    def verbalize(self, input):
        lattice = input @ self.verbalizer
        return shortestpath(lattice, nshortest=1, unique=True).string()

    def normalize(self, input):
        output = escape(input)
        output = self.tag(output)
        output = self.parser.reorder(output)
        output = escape(output)
        return self.verbalize(output)
