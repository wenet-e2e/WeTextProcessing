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


TOKEN_ORDER = {"electronic": ["protocol", "username", "domain"]}


class Electronic(Processor):
    def __init__(self, tn_electronic):
        super().__init__("electronic", ordertype="itn", token_orders=TOKEN_ORDER)
        self.username_graph = tn_electronic.username_graph.copy().invert().optimize()
        self.domain_graph = tn_electronic.domain_graph.copy().invert().optimize()
        self.protocol_graph = tn_electronic.protocol_graph.copy().invert().optimize()
        non_period = (self.VCHAR - ".").plus
        social = non_period + (pynini.accep(".") + non_period).star + pynini.accep(".").ques
        social_graph = (social @ tn_electronic.username_graph).invert().optimize()

        domain_tagger = self.tag_field("domain", self.domain_graph)
        domain_verbalizer = self.verbalize_field("domain", self.domain_graph)
        email_tagger = (
            self.tag_field("username", self.username_graph) + pynutil.delete(" at ")
            + pynutil.insert(" ") + domain_tagger
        )
        email_verbalizer = (
            self.verbalize_field("username", self.username_graph) + pynutil.delete(" ")
            + pynutil.insert("@") + domain_verbalizer
        )
        social_tagger = pynutil.delete("at ") + self.tag_field("username", social_graph)
        social_verbalizer = pynutil.insert("@") + self.verbalize_field("username", social_graph)
        url_tagger = (
            self.tag_field("protocol", self.protocol_graph) + pynini.accep(" ") + domain_tagger
        )
        url_verbalizer = (
            self.verbalize_field("protocol", self.protocol_graph) + pynutil.delete(" ") + domain_verbalizer
        )
        self.graph = (
            self.domain_graph
            | self.username_graph + pynutil.delete(" at ") + pynutil.insert("@") + self.domain_graph
            | pynutil.delete("at ") + pynutil.insert("@") + social_graph
            | self.protocol_graph + pynutil.delete(" ") + self.domain_graph
        ).optimize()
        self.fields_tagger = (domain_tagger | email_tagger | social_tagger | url_tagger).optimize()
        self.fields_verbalizer = (domain_verbalizer | email_verbalizer | social_verbalizer | url_verbalizer).optimize()
        self.tagger = self.add_tokens(self.fields_tagger)
        self.verbalizer = self.delete_tokens(self.fields_verbalizer)
