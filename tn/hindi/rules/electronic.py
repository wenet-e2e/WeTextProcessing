# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import pynini
from pynini.lib.pynutil import add_weight, delete, insert

from tn.hindi.rules.utils import HI_DIGIT, hindi_path, squeeze_spaces
from tn.processor import Processor
from tn.utils import load_labels

TOKEN_ORDER = {}


class Electronic(Processor):
    """E-mails, URLs, paths, IPs and chemical formulas keep their written text raw.

    ``kumar@gmail.com`` -> ``electronic { username: "kumar" domain: "gmail.com" }``
    -> ``kumar एट gmail डॉट com``: Latin runs stay verbatim, digits and symbols are read in Hindi.
    """

    def __init__(self):
        super().__init__("electronic", token_orders=TOKEN_ORDER)
        subscript = pynini.string_file(hindi_path("electronic/subscript_digit.tsv"))
        subscript_digit = subscript.copy().project("input")
        alphanumeric = self.ALPHA | self.DIGIT | HI_DIGIT | subscript_digit

        classes = {"email": [], "url": [], "unix": [], "windows": [], "chem": []}
        for row in load_labels(hindi_path("electronic/symbol_classes.tsv")):
            if len(row) == 2:
                for name in row[1].split(","):
                    if name in classes:
                        classes[name].append(row[0])
        email_symbols = pynini.union(*classes["email"])
        url_symbols = pynini.union(*classes["url"])
        unix_symbols = pynini.union(*classes["unix"])
        windows_symbols = pynini.union(*classes["windows"])
        chem_symbols = pynini.union(*classes["chem"])
        unix_segment_symbols = pynini.union(*[s for s in classes["unix"] if s != "/"])

        username = (self.ALPHA | self.DIGIT | email_symbols).plus
        domain = (self.ALPHA | self.DIGIT | "." | "-").plus

        protocol_start = pynini.union("https://", "http://")
        protocol = pynini.union(add_weight(protocol_start + "www.", 1.0), add_weight(protocol_start, 1.01),
                                add_weight(pynini.accep("www."), 1.02))
        url_domain = (alphanumeric | url_symbols).plus

        windows_path = self.ALPHA + ":" + "\\" + (alphanumeric | windows_symbols | " ").plus
        unix_path = (pynini.accep("/") + (alphanumeric | unix_symbols).plus
                     | (alphanumeric | unix_segment_symbols).plus + "/" + (alphanumeric | unix_symbols).star)
        backslash_path = pynini.accep("\\") + (alphanumeric | unix_segment_symbols | "\\" | " ").plus
        octet = pynini.closure(self.DIGIT, 1, 3)
        ip_address = octet + ("." + octet)**3
        segment = (self.ALPHA | self.DIGIT | "-").plus
        tld = pynini.string_file(hindi_path("electronic/domain.tsv")).project("input")
        domain_body = (segment + ".").plus + tld + ("." + tld).ques
        combined_domain = domain_body + pynini.accep("/").ques
        extensions = pynini.string_file(hindi_path("electronic/file_extensions.tsv")).project("input")
        file_name = (self.ALPHA | self.DIGIT | "-" | "_").plus + "." + extensions

        elements = pynini.string_file(hindi_path("electronic/elements.tsv")).project("input")
        chem_number = (self.DIGIT | subscript_digit).plus
        chem_block = elements + chem_number.ques
        chem_chars = chem_block | chem_symbols | chem_number
        raw_chemical = chem_symbols.star + chem_block + chem_chars.star
        any_chem = chem_chars.star
        no_open = pynini.difference(any_chem, any_chem + "(" + any_chem)
        unbalanced = pynini.intersect(no_open, any_chem + ")")
        chemical = pynini.difference(raw_chemical, unbalanced)
        chem_sigma = (self.ALPHA | self.DIGIT | subscript_digit | chem_symbols).star
        chemical = pynini.intersect(chemical, chem_sigma + subscript_digit + chem_sigma).optimize()

        # Verbalizer content: Latin runs verbatim, every other character read out.
        symbols = pynini.string_file(hindi_path("electronic/symbols.tsv"))
        digits = pynini.union(
            pynini.string_file(hindi_path("telephone/number.tsv")),
            pynini.string_file(hindi_path("numbers/digit.tsv")),
            pynini.string_file(hindi_path("numbers/zero.tsv")), subscript)
        single_digit = digits + insert(" ")
        single_symbol = symbols + insert(" ")
        non_alpha = add_weight(single_symbol, 1.0) | add_weight(single_digit, 1.0)
        alpha_run = self.ALPHA.plus + insert(" ")
        content = (non_alpha.star + (alpha_run + non_alpha.plus).star + alpha_run.ques + non_alpha.star)
        letters = pynini.string_file(hindi_path("address/letters.tsv"))
        letters = letters | ((self.TO_LOWER + self.VSIGMA) @ letters)
        chem_content = ((letters + insert(" ")) | single_digit | single_symbol).plus
        ip_content = (single_symbol | single_digit).plus
        protocols = pynini.string_file(hindi_path("electronic/protocols.tsv"))
        protocol_reading = pynini.union(
            pynini.cross("https://www.", "httpswww"), pynini.cross("http://www.", "httpwww"),
            pynini.cross("https://", "https"), pynini.cross("http://", "http"),
            pynini.cross("www.", "www")) @ protocols

        squeeze = squeeze_spaces()

        def field(name, raw, reading, first=True):
            """The tagger keeps ``raw`` (with NeMo's classifier weights); the verbalizer reads it."""

            spoken = (raw @ reading @ squeeze).optimize()
            tagger = self.tag_field(name, raw)
            verbalizer = self.verbalize_field(name, spoken, leading_space=not first)
            if not first:
                tagger = insert(" ") + tagger
                verbalizer = insert(" ") + verbalizer
            return tagger, verbalizer

        user_t, user_v = field("username", username, content)
        email_domain_t, email_domain_v = field("domain", domain, content, first=False)
        protocol_t, protocol_v = field("protocol", protocol, protocol_reading)
        url_domain_t, url_domain_v = field("domain", url_domain, content, first=False)
        path_t, path_v = field("path", windows_path | unix_path | backslash_path, content)
        ip_t, ip_v = field("domain", ip_address, ip_content)
        domain_t, domain_v = field("domain", combined_domain | file_name | ip_address, content)
        chem_t, chem_v = field("fragment_id", chemical, chem_content)

        taggers = [
            (protocol_t + url_domain_t, 1.0),
            (user_t + delete("@") + email_domain_t, 1.0),
            (path_t, 1.0),
            (ip_t, 1.0),
            (field("domain", combined_domain | file_name, content)[0], 1.1),
            (chem_t, 1.2),
        ]
        verbalizers = [
            (protocol_v + url_domain_v, 1.0),
            (user_v + insert(" एट") + email_domain_v, 1.01),
            (path_v, 1.02),
            (ip_v, 1.03),
            (domain_v, 1.04),
            (chem_v, 1.04),
        ]
        self.tagger = self.add_tokens(pynini.union(*[add_weight(g, w) for g, w in taggers]))
        self.verbalizer = self.delete_tokens(pynini.union(*[add_weight(g, w) for g, w in verbalizers]))
