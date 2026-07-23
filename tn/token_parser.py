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

EOS = "<EOS>"
TN_ORDERS = {
    "date": ["year", "month", "day"],
    "fraction": ["denominator", "numerator"],
    "measure": ["denominator", "numerator", "value"],
    "money": ["value", "currency"],
    "time": ["noon", "hour", "minute", "second"],
}
EN_TN_ORDERS = {
    "date": ["preserve_order", "text", "day", "month", "year"],
    "money": ["integer_part", "fractional_part", "quantity", "currency_maj"],
}
ITN_ORDERS = {
    "date": ["year", "month", "day", "preserve_order"],
    "fraction": ["sign", "numerator", "denominator"],
    "measure": ["numerator", "denominator", "value", "units"],
    "money": ["currency", "value", "decimal", "quantity"],
    "time": ["hour", "minute", "second", "noon", "zone"],
    "telephone": ["country_code", "number_part"],
    "electronic": ["username", "domain", "protocol"],
}


class TokenParseError(ValueError):
    """Raised when a tagged token stream is malformed."""


class Token:

    def __init__(self, name):
        self.name = name
        self.order = []
        self.members = {}

    def append(self, key, value):
        self.order.append(key)
        self.members[key] = value

    def string(self, orders):
        output = self.name + " {"
        if self.name in orders.keys():
            if "preserve_order" not in self.members.keys() or self.members["preserve_order"] != "true":
                self.order = orders[self.name]

        for key in self.order:
            if key not in self.members.keys():
                continue
            output += ' {}: "{}"'.format(key, self.members[key])
        return output + " }"


class TokenParser:

    def __init__(self, ordertype="tn"):
        if ordertype == "tn":
            self.orders = TN_ORDERS
        elif ordertype == "itn":
            self.orders = ITN_ORDERS
        elif ordertype == "en_tn":
            self.orders = EN_TN_ORDERS
        else:
            raise NotImplementedError()

    def load(self, input):
        if not input:
            raise TokenParseError("token stream must not be empty")
        self.index = 0
        self.text = input
        self.char = input[0]
        self.tokens = []

    def read(self):
        if self.index < len(self.text) - 1:
            self.index += 1
            self.char = self.text[self.index]
            return True
        self.char = EOS
        return False

    def parse_ws(self):
        not_eos = self.char != EOS
        while not_eos and self.char == " ":
            not_eos = self.read()
        return not_eos

    def parse_char(self, exp):
        if self.char == exp:
            self.read()
            return True
        return False

    def parse_chars(self, exp):
        start = self.index
        for x in exp:
            if not self.parse_char(x):
                self.index = start
                self.char = self.text[start]
                return False
        return True

    def expect_chars(self, exp):
        if not self.parse_chars(exp):
            raise TokenParseError(
                'expected {!r} at position {}, got {!r}'.format(exp, self.index, self.char)
            )

    def parse_key(self):
        if self.char == EOS:
            raise TokenParseError("expected key at end of token stream")
        if self.char in string.whitespace:
            raise TokenParseError("expected key at position {}".format(self.index))

        key = ""
        while self.char in string.ascii_letters + "_":
            key += self.char
            self.read()
        if not key:
            raise TokenParseError("invalid key at position {}".format(self.index))
        return key

    def parse_value(self):
        if self.char == EOS:
            raise TokenParseError("expected value at end of token stream")

        value = ""
        while self.char != '"':
            if self.char == EOS:
                raise TokenParseError("unterminated value at position {}".format(self.index))
            value += self.char
            escaped = self.char == "\\"
            self.read()
            if escaped:
                if self.char == EOS:
                    raise TokenParseError("unterminated escape at position {}".format(self.index))
                value += self.char
                self.read()
        return value

    def parse(self, input):
        self.load(input)
        while self.parse_ws():
            name = self.parse_key()
            self.expect_chars(" { ")

            token = Token(name)
            closed = False
            while self.parse_ws():
                if self.char == "}":
                    self.parse_char("}")
                    closed = True
                    break
                key = self.parse_key()
                self.expect_chars(': "')
                value = self.parse_value()
                self.expect_chars('"')
                token.append(key, value)
            if not closed:
                raise TokenParseError("unterminated token {!r}".format(name))
            self.tokens.append(token)

    def reorder(self, input):
        self.parse(input)
        output = ""
        for token in self.tokens:
            output += token.string(self.orders) + " "
        return output.strip()
