# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pynini
from pynini.lib import pynutil

from tn.processor import Processor
from tn.utils import get_abs_path, load_labels


TOKEN_ORDER = {}


class Electronic(Processor):
    def __init__(self, deterministic=True):
        super().__init__("electronic", token_orders=TOKEN_ORDER)
        self.deterministic = deterministic
        symbols = pynini.string_file(get_abs_path("german/data/electronic/symbols.tsv"))
        symbol_chars = self.input_projection(symbols)
        non_period = (self.ALPHA | self.DIGIT | (symbol_chars - ".")).plus
        username = (self.ALPHA | self.DIGIT | symbol_chars).plus
        domain = non_period + (pynini.accep(".") + non_period).plus + pynini.accep(".").ques
        protocol_start = pynini.union("http://", "https://")
        protocol = protocol_start | "www." | (protocol_start + "www.")

        digits = pynini.string_file(get_abs_path("german/data/numbers/digit.tsv")).invert()
        digits = digits | pynini.cross("1", "eins")
        digits = digits | pynini.string_file(get_abs_path("german/data/numbers/zero.tsv")).invert()
        expand = pynini.cdrewrite(symbols | digits, "", "", self.VSIGMA)

        def spell(language):
            char = self.VCHAR - pynini.union('"', " ")
            return (language @ ((char + pynutil.insert(" ")).star + char) @ expand).optimize()

        # Bare input-to-spoken components; no serialized token fields.
        self.username_graph = spell(username)
        self.protocol_graph = spell(protocol)
        servers = pynini.string_map(load_labels(get_abs_path("german/data/electronic/server_name.tsv")))
        suffixes = pynini.string_file(get_abs_path("german/data/electronic/domain.tsv"))
        default = pynutil.add_weight(self.VCHAR - '"', 0.0001) | servers | suffixes
        final_period = pynini.cdrewrite(pynini.cross(" punkt", "."), "", "[EOS]", self.VSIGMA)
        self.domain_graph = (domain @ (default + (pynutil.insert(" ") + default).star) @ expand).optimize()

        domain_tagger = self.tag_field("domain", self.domain_graph)
        domain_verbalizer = self.verbalize_field("domain", self.domain_graph)
        email_tagger = self.tag_field("username", self.username_graph) + pynutil.delete("@")
        email_tagger = email_tagger + pynutil.insert(" ") + domain_tagger
        email_verbalizer = self.verbalize_field("username", self.username_graph)
        email_verbalizer = email_verbalizer + pynutil.delete(" ") + pynutil.insert(" at ") + domain_verbalizer
        social = (non_period + (pynini.accep(".") + non_period).star + pynini.accep(".").ques).optimize()
        social_tagger = pynutil.delete("@") + self.tag_field("username", social @ self.username_graph)
        social_verbalizer = pynutil.insert("at ") + self.verbalize_field("username", self.username_graph)
        url_tagger = self.tag_field("protocol", self.protocol_graph) + pynutil.insert(" ") + domain_tagger
        url_verbalizer = self.verbalize_field("protocol", self.protocol_graph)
        url_verbalizer = url_verbalizer + pynutil.delete(" ") + pynutil.insert(" ") + domain_verbalizer
        self.fields_tagger = (domain_tagger | email_tagger | social_tagger | url_tagger).optimize()
        self.fields_verbalizer = (domain_verbalizer | email_verbalizer | social_verbalizer | url_verbalizer).optimize()
        self.tagger = self.add_tokens(self.fields_tagger)
        self.verbalizer = (self.delete_tokens(self.fields_verbalizer) @ final_period).optimize()
