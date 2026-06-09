# Copyright (c) 2026 Zhendong Peng (pzd17@tsinghua.org.cn)
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

from pynini import closure, cross, invert, string_file
from pynini.lib.pynutil import add_weight, delete, insert

from tn.processor import Processor
from tn.utils import get_abs_path


class Electronic(Processor):

    def __init__(self):
        super().__init__(name="electronic", ordertype="itn")
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        ds = delete(" ")

        # Single characters: digits and letters
        digit = string_file(get_abs_path("../itn/english/data/numbers/digit.tsv"))
        zero = string_file(get_abs_path("../itn/english/data/numbers/zero.tsv"))
        alpha_or_digit = self.ALPHA | digit | zero

        # Symbols from TSV (symbol\tname): invert to get name -> symbol
        symbols = invert(
            string_file(get_abs_path("../itn/english/data/electronic/symbols.tsv"))
        )

        # A "token" is either a single char (letter/digit/symbol) or a
        # multi-letter word kept verbatim (e.g. "gmail", "nvidia").
        # Multi-letter words have lower priority so spelled-out letters are preferred.
        word = add_weight(closure(self.ALPHA, 2), 0.01)
        token = alpha_or_digit | symbols | word

        # A component is one or more tokens separated by spaces
        component = token + closure(ds + token)

        username = insert('username: "') + component + insert('"')

        # Domain: component(s) separated by "dot" => "."
        dot = cross("dot", ".")
        domain_content = component + closure(ds + dot + ds + component)
        domain = insert('domain: "') + domain_content + insert('"')

        # Email: username at domain
        graph_email = (
            username
            + ds
            + delete("at")
            + ds
            + insert(" ")
            + domain
        )

        # URL protocol: "h t t p colon slash slash" or "h t t p s colon slash slash"
        http = cross("h t t p", "http")
        https = cross("h t t p s", "https")
        colon_slash_slash = cross(" colon slash slash ", "://")
        protocol_start = (http | https) + colon_slash_slash

        # www prefix
        www = cross("w w w", "www")

        # URL: [protocol] [www.] domain
        url_content = (
            closure(protocol_start, 0, 1)
            + closure(www + ds + dot + ds, 0, 1)
            + domain_content
        )
        graph_url = insert('protocol: "') + url_content + insert('"')

        final_graph = graph_email | graph_url
        self.tagger = self.add_tokens(final_graph)

    def build_verbalizer(self):
        username = (
            delete("username:")
            + self.DELETE_SPACE
            + delete('"')
            + self.NOT_QUOTE.plus
            + delete('"')
        )
        domain = (
            delete("domain:")
            + self.DELETE_SPACE
            + delete('"')
            + self.NOT_QUOTE.plus
            + delete('"')
        )
        protocol = (
            delete("protocol:")
            + self.DELETE_SPACE
            + delete('"')
            + self.NOT_QUOTE.plus
            + delete('"')
        )

        # Email: username@domain
        graph_email = username + self.DELETE_SPACE + insert("@") + domain
        # URL: just output the protocol content directly
        graph_url = protocol

        graph = graph_email | graph_url
        self.verbalizer = self.delete_tokens(graph)
