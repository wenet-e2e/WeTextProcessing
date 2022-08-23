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

import string
from itertools import permutations, product

EOS = '<EOS>'


class TokenParser:

    def __call__(self, text):
        self.text = text
        assert len(text) > 0
        self.char = text[0]
        self.index = 0

    def read(self):
        if self.index < len(self.text) - 1:
            self.index += 1
            self.char = self.text[self.index]
            return True
        self.char = EOS
        return False

    def parse_ws(self):
        not_eos = self.char != EOS
        while not_eos and self.char == ' ':
            not_eos = self.read()
        return not_eos

    def parse_char(self, exp):
        if self.char == exp:
            self.read()
            return True
        return False

    def parse_chars(self, exp):
        ok = False
        for x in exp:
            ok |= self.parse_char(x)
        return ok

    def parse_key(self):
        assert self.char != EOS
        assert self.char not in string.whitespace

        chars = []
        while self.char in string.ascii_letters + '_':
            chars.append(self.char)
            self.read()
        return ''.join(chars)

    def parse_value(self):
        assert self.char != EOS
        chars = []
        while self.char != '"':
            chars.append(self.char)
            self.read()
        return ''.join(chars)

    def parse(self):
        tokens = []
        while self.parse_ws():
            name = self.parse_key()
            self.parse_chars(' { ')

            sub_toks = []
            while self.parse_ws():
                if self.char == '}':
                    self.parse_char('}')
                    break
                key = self.parse_key()
                self.parse_chars(': "')
                value = self.parse_value()
                self.parse_chars('"')
                sub_toks.append((key, value))
            tokens.append((name, sub_toks))

        return tokens

    def permute(self):

        def helper(key, pairs):
            text = key + ' {'
            for k, v in pairs:
                text += ' {}: "{}"'.format(k, v)
            return text + ' } '

        tokens = self.parse()
        prefixes = ['']
        for k, v in tokens:
            values = [helper(k, pairs) for pairs in permutations(v)]
            prefixes = [''.join(x) for x in product(prefixes, values)]
        return [prefix.strip() for prefix in prefixes]
