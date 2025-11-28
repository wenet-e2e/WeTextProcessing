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

from tn.token_parser import EOS, TokenParser


class TestTokenParser:

    parser = TokenParser()

    def test_read(self):
        self.parser.load(" ")
        assert self.parser.read() is False
        assert self.parser.char == EOS

    def test_parse_ws(self):
        self.parser.load(" ")
        assert self.parser.parse_ws() is False
        assert self.parser.char == EOS

        self.parser.load("  ")
        assert self.parser.parse_ws() is False
        assert self.parser.char == EOS

        self.parser.load("  test")
        assert self.parser.parse_ws() is True
        assert self.parser.char == "t"

    def test_parse_chars(self):
        self.parser.load("hello world")
        assert self.parser.parse_chars("hello") is True
        assert self.parser.char == " "

        self.parser.load("world")
        assert self.parser.parse_chars("hello") is False
        assert self.parser.char == "w"

    def test_parse_key(self):
        self.parser.load("key")
        assert self.parser.parse_key() == "key"

        self.parser.load("key ")
        assert self.parser.parse_key() == "key"

    def test_parse_value(self):
        self.parser.load('value"')
        assert self.parser.parse_value() == "value"

    def test_parse(self):
        input = 'time { minute: "零二分" hour: "两点" } char { value: "走" }'
        self.parser.parse(input)
        tokens = self.parser.tokens

        assert len(tokens) == 2
        assert tokens[0].name == "time"
        assert tokens[1].name == "char"
        assert tokens[0].order == ["minute", "hour"]
        assert tokens[1].order == ["value"]
        assert tokens[0].members == {"minute": "零二分", "hour": "两点"}
        assert tokens[1].members == {"value": "走"}

    def test_reorder(self):
        input = 'time { minute: "零二分" hour: "两点" } char { value: "走" }'
        expected = 'time { hour: "两点" minute: "零二分" } char { value: "走" }'
        assert self.parser.reorder(input) == expected
