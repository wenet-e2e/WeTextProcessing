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

from pynini import accep, closure, cross, difference, invert, string_file
from pynini.lib.pynutil import add_weight, delete, insert

from tn.processor import Processor
from itn.utils import get_abs_path


class Electronic(Processor):

    def __init__(self):
        super().__init__(name="electronic", ordertype="itn")
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        ds = delete(" ")
        digit = string_file(get_abs_path("english/data/numbers/digit.tsv"))
        zero = string_file(get_abs_path("english/data/numbers/zero.tsv"))
        symbols = invert(string_file(get_abs_path("english/data/electronic/symbols.tsv")))

        char = self.ALPHA | digit | zero
        word = add_weight(closure(self.ALPHA, 2), 0.1)
        token = char | symbols | word
        first_token = char | difference(word, accep("dot"))
        component = first_token + closure(ds + token)

        dot = cross("dot", ".")
        domain = component + (ds + dot + ds + component).plus

        self.username = component
        self.domain = delete("at") + ds + domain
        username = self.tag_field("username", self.username)
        domain_field = self.tag_field("domain", self.domain)

        # Email: X at Y dot Z (requires "at" keyword)
        graph_email = add_weight(username + ds + insert(" ") + domain_field, -0.001)

        # URL: requires protocol or www prefix
        http = cross("h t t p", "http")
        https = cross("h t t p s", "https")
        protocol = (http | https) + cross(" colon slash slash ", "://")
        www = cross("w w w", "www")

        # protocol + [www.] + domain
        url_with_protocol = protocol + closure(www + ds + dot + ds, 0, 1) + domain
        # www. + domain (no protocol)
        url_with_www = www + ds + dot + ds + domain
        # domain only (must have dot): nvidia dot com
        url_domain_only = domain

        self.protocol = url_with_protocol | url_with_www | url_domain_only
        graph_url = self.tag_field("protocol", self.protocol)

        final_graph = graph_email | graph_url
        self.tagger = self.add_tokens(final_graph)

    def build_verbalizer(self):
        username = self.verbalize_field("username", self.username)
        domain = self.verbalize_field("domain", self.domain)
        protocol = self.verbalize_field("protocol", self.protocol)

        graph_email = username + self.DELETE_SPACE + insert("@") + domain
        graph_url = protocol

        self.verbalizer = self.delete_tokens(graph_email | graph_url)
