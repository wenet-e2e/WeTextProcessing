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

import logging
import os
import string

from pynini import Fst, cdrewrite, closure, cross, difference, escape, invert, shortestpath, union
from pynini.lib import byte, utf8
from pynini.lib.pynutil import delete, insert

from tn.token_parser import TokenParser


class Processor:

    def __init__(self, name, ordertype="tn"):
        self.ALPHA = byte.ALPHA
        self.DIGIT = byte.DIGIT
        self.PUNCT = byte.PUNCT
        self.SPACE = byte.SPACE | u'\u00A0'
        self.VCHAR = utf8.VALID_UTF8_CHAR
        self.VSIGMA = self.VCHAR.star
        self.LOWER = byte.LOWER
        self.UPPER = byte.UPPER

        CHAR = difference(self.VCHAR, union('\\', '"'))
        self.CHAR = (CHAR | cross('\\', '\\\\\\') | cross('"', '\\"'))
        self.SIGMA = (CHAR | cross('\\\\\\', '\\') | cross('\\"', '"')).star
        self.NOT_QUOTE = difference(self.VCHAR, r'"').optimize()
        self.NOT_SPACE = difference(self.VCHAR, self.SPACE).optimize()
        self.INSERT_SPACE = insert(" ")
        self.DELETE_SPACE = delete(self.SPACE).star
        self.DELETE_EXTRA_SPACE = cross(self.SPACE.plus, " ")
        self.DELETE_ZERO_OR_ONE_SPACE = delete(self.SPACE.ques)
        self.MIN_NEG_WEIGHT = -0.0001
        self.TO_LOWER = union(*[
            cross(x, y)
            for x, y in zip(string.ascii_uppercase, string.ascii_lowercase)
        ])
        self.TO_UPPER = invert(self.TO_LOWER)

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
        logger = logging.getLogger('wetext-{}'.format(self.name))
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        fmt = logging.Formatter('%(asctime)s WETEXT %(levelname)s %(message)s')
        handler.setFormatter(fmt)
        logger.addHandler(handler)

        os.makedirs(cache_dir, exist_ok=True)
        tagger_name = '{}_tagger.fst'.format(prefix)
        verbalizer_name = '{}_verbalizer.fst'.format(prefix)

        tagger_path = os.path.join(cache_dir, tagger_name)
        verbalizer_path = os.path.join(cache_dir, verbalizer_name)

        exists = os.path.exists(tagger_path) and os.path.exists(
            verbalizer_path)
        if exists and not overwrite_cache:
            logger.info("found existing fst: {}".format(tagger_path))
            logger.info("                    {}".format(verbalizer_path))
            logger.info("skip building fst for {} ...".format(self.name))
            self.tagger = Fst.read(tagger_path).optimize()
            self.verbalizer = Fst.read(verbalizer_path).optimize()
        else:
            logger.info("building fst for {} ...".format(self.name))
            self.build_tagger()
            self.build_verbalizer()
            self.tagger.optimize().write(tagger_path)
            self.verbalizer.optimize().write(verbalizer_path)
            logger.info("done")
            logger.info("fst path: {}".format(tagger_path))
            logger.info("          {}".format(verbalizer_path))

    def tag(self, input):
        if len(input) == 0:
            return ''
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
