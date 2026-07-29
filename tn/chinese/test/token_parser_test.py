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

import pytest

from tn.token_parser import EOS, TokenParseError, TokenParser


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

    def test_parse_value_decodes_wire_escapes(self):
        self.parser.load('中\\"文\\\\尾"')
        assert self.parser.parse_value() == '中"文\\尾'

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

    def test_reorder_preserves_unknown_fields(self):
        input = 'money { unexpected: "keep" currency: "元" value: "十" }'
        expected = 'money { value: "十" currency: "元" unexpected: "keep" }'
        assert self.parser.reorder(input) == expected

    def test_reorder_round_trips_escaped_unicode_value(self):
        tagged = 'char { value: "中\\"文\\\\尾" }'

        self.parser.parse(tagged)
        assert self.parser.tokens[0].members["value"] == '中"文\\尾'
        assert self.parser.reorder(tagged) == tagged

    def test_duplicate_field_is_rejected(self):
        with pytest.raises(TokenParseError, match="duplicate field"):
            self.parser.reorder('char { value: "first" value: "second" }')

    @pytest.mark.parametrize(
        "tagged",
        [
            'time { hour: "12 }',
            'time { hour "12" }',
            'time { hour: "12" ',
        ],
    )
    def test_malformed_input(self, tagged):
        with pytest.raises(TokenParseError):
            self.parser.reorder(tagged)
