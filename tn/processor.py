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

import os

from tn.token_parser import TokenParser

from pynini import cdrewrite, cross, difference, escape, Fst, shortestpath, union
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
        self.ordertype = ordertype
        self.tagger = None
        self.verbalizer = None

    def build_rule(self, fst, l='', r=''):
        rule = cdrewrite(fst, l, r, self.VSIGMA)
        return rule

    def add_tokens(self, tagger):
        tagger = insert(f"{self.name} {{ ") + tagger + insert(' } ')
        return tagger.optimize()

    def delete_tokens(self, verbalizer):
        verbalizer = (delete(f"{self.name}") + delete(' { ') + verbalizer +
                      delete(' }') + delete(' ').ques)
        return verbalizer.optimize()

    def build_verbalizer(self):
        verbalizer = delete('value: "') + self.SIGMA + delete('"')
        self.verbalizer = self.delete_tokens(verbalizer)

    def build_fst(self, prefix, cache_dir, overwrite_cache):
        os.makedirs(cache_dir, exist_ok=True)
        tagger_name = '{}_tagger.fst'.format(prefix)
        verbalizer_name = '{}_verbalizer.fst'.format(prefix)

        tagger_path = os.path.join(cache_dir, tagger_name)
        verbalizer_path = os.path.join(cache_dir, verbalizer_name)

        exists = os.path.exists(tagger_path) and os.path.exists(
            verbalizer_path)
        if exists and not overwrite_cache:
            self.tagger = Fst.read(tagger_path).optimize()
            self.verbalizer = Fst.read(verbalizer_path).optimize()
        else:
            self.build_tagger()
            self.build_verbalizer()
            self.tagger.optimize().write(tagger_path)
            self.verbalizer.optimize().write(verbalizer_path)

    def tag(self, input):
        input = escape(input)
        lattice = input @ self.tagger
        return shortestpath(lattice, nshortest=1, unique=True).string()

    def verbalize(self, input):
        # Only words from the blacklist are contained.
        if len(input) == 0:
            return ''
        output = TokenParser(self.ordertype).reorder(input)
        # We need escape for pynini to build the fst from string.
        lattice = escape(output) @ self.verbalizer
        return shortestpath(lattice, nshortest=1, unique=True).string()

    def normalize(self, input):
        return self.verbalize(self.tag(input))
