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

import hashlib
import json
import logging
import os
import string

from pynini import Fst, cdrewrite, cross, difference, escape, invert, shortestpath, union
from pynini.lib import byte, utf8
from pynini.lib.pynutil import delete, insert

from tn.token_parser import TokenParser

logger = logging.getLogger("wetext")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s WETEXT %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class Processor:

    def __init__(self, name, ordertype="tn"):
        self.ALPHA = byte.ALPHA
        self.DIGIT = byte.DIGIT
        self.PUNCT = byte.PUNCT
        self.SPACE = byte.SPACE | "\u00A0"
        self.VCHAR = utf8.VALID_UTF8_CHAR
        self.VSIGMA = self.VCHAR.star
        self.LOWER = byte.LOWER
        self.UPPER = byte.UPPER

        CHAR = difference(self.VCHAR, union("\\", '"'))
        self.CHAR = CHAR | cross("\\", "\\\\\\") | cross('"', '\\"')
        self.SIGMA = (CHAR | cross("\\\\\\", "\\") | cross('\\"', '"')).star
        self.NOT_QUOTE = difference(self.VCHAR, r'"').optimize()
        self.NOT_SPACE = difference(self.VCHAR, self.SPACE).optimize()
        self.INSERT_SPACE = insert(" ")
        self.DELETE_SPACE = delete(self.SPACE).star
        self.DELETE_EXTRA_SPACE = cross(self.SPACE.plus, " ")
        self.DELETE_ZERO_OR_ONE_SPACE = delete(self.SPACE.ques)
        self.MIN_NEG_WEIGHT = -0.0001
        self.TO_LOWER = union(*[cross(x, y) for x, y in zip(string.ascii_uppercase, string.ascii_lowercase)])
        self.TO_UPPER = invert(self.TO_LOWER)

        self.name = name
        self.ordertype = ordertype
        self.tagger = None
        self.verbalizer = None

    def build_rule(self, fst, l="", r=""):
        rule = cdrewrite(fst, l, r, self.VSIGMA)
        return rule

    def add_tokens(self, tagger):
        tagger = insert(f"{self.name} {{ ") + tagger + insert(" } ")
        return tagger.optimize()

    def delete_tokens(self, verbalizer):
        verbalizer = delete(f"{self.name}") + delete(" { ") + verbalizer + delete(" }") + delete(" ").ques
        return verbalizer.optimize()

    def build_verbalizer(self):
        verbalizer = delete('value: "') + self.SIGMA + delete('"')
        self.verbalizer = self.delete_tokens(verbalizer)

    @staticmethod
    def _source_fingerprint(prefix):
        language_code, mode = prefix.split("_", 1)
        language = {"en": "english", "ja": "japanese", "zh": "chinese"}[language_code]
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        package_root = os.path.join(project_root, mode)
        sources = [
            os.path.join(project_root, "tn", "processor.py"),
            os.path.join(project_root, "tn", "token_parser.py"),
            os.path.join(project_root, "tn", "utils.py"),
        ]
        language_root = os.path.join(package_root, language)
        entrypoint = "normalizer.py" if mode == "tn" else "inverse_normalizer.py"
        sources.append(os.path.join(language_root, entrypoint))
        source_roots = [
            os.path.join(language_root, "data"),
            os.path.join(language_root, "rules"),
        ]
        for source_root in source_roots:
            for root, _, filenames in os.walk(source_root):
                for filename in filenames:
                    if filename.endswith((".py", ".tsv", ".far")):
                        sources.append(os.path.join(root, filename))

        digest = hashlib.sha256()
        for source in sorted(sources):
            relative_path = os.path.relpath(source, project_root)
            digest.update(relative_path.encode("utf-8"))
            with open(source, "rb") as source_file:
                for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                    digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _cache_config_matches(manifest_path, prefix, cache_config, source_fingerprint):
        try:
            with open(manifest_path, encoding="utf-8") as manifest_file:
                manifest = json.load(manifest_file)
        except (OSError, ValueError):
            return False
        return (
            manifest.get("version") == 1
            and manifest.get("prefix") == prefix
            and manifest.get("config") == cache_config
            and manifest.get("source_fingerprint") == source_fingerprint
        )

    @staticmethod
    def _write_cache_manifest(manifest_path, prefix, cache_config, source_fingerprint):
        manifest = {
            "version": 1,
            "prefix": prefix,
            "config": cache_config,
            "source_fingerprint": source_fingerprint,
        }
        temporary_path = "{}.tmp.{}".format(manifest_path, os.getpid())
        with open(temporary_path, "w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file, ensure_ascii=False, sort_keys=True)
            manifest_file.write("\n")
        os.replace(temporary_path, manifest_path)

    def build_fst(self, prefix, cache_dir, overwrite_cache, cache_config=None):
        cache_dir = os.fspath(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        cache_config = {} if cache_config is None else cache_config
        source_fingerprint = self._source_fingerprint(prefix)
        tagger_name = "{}_tagger.fst".format(prefix)
        verbalizer_name = "{}_verbalizer.fst".format(prefix)
        manifest_name = "{}_cache.json".format(prefix)

        tagger_path = os.path.join(cache_dir, tagger_name)
        verbalizer_path = os.path.join(cache_dir, verbalizer_name)
        manifest_path = os.path.join(cache_dir, manifest_name)

        exists = os.path.exists(tagger_path) and os.path.exists(verbalizer_path)
        cache_matches = self._cache_config_matches(
            manifest_path,
            prefix,
            cache_config,
            source_fingerprint,
        )
        if exists and cache_matches and not overwrite_cache:
            logger.info("found existing fst: {}".format(tagger_path))
            logger.info("                    {}".format(verbalizer_path))
            logger.info("skip building fst for {} ...".format(self.name))
            self.tagger = Fst.read(tagger_path).optimize()
            self.verbalizer = Fst.read(verbalizer_path).optimize()
        else:
            logger.info("building fst for {} ...".format(self.name))
            if hasattr(self, 'build_tagger_and_verbalizer'):
                self.build_tagger_and_verbalizer()
            else:
                self.build_tagger()
                self.build_verbalizer()
            self.tagger.optimize().write(tagger_path)
            self.verbalizer.optimize().write(verbalizer_path)
            self._write_cache_manifest(
                manifest_path,
                prefix,
                cache_config,
                source_fingerprint,
            )
            logger.info("done")
            logger.info("fst path: {}".format(tagger_path))
            logger.info("          {}".format(verbalizer_path))

    def tag(self, input, nbest=1):
        if len(input) == 0:
            return "" if nbest == 1 else [""]
        input = escape(input)
        lattice = input @ self.tagger
        if nbest == 1:
            return shortestpath(lattice, nshortest=1, unique=True).string()
        lattice = shortestpath(lattice.project("output").rmepsilon(), nshortest=nbest, unique=True)
        paths = lattice.paths()
        results = []
        while not paths.done():
            results.append(paths.ostring())
            paths.next()
        return results

    def verbalize(self, input):
        if len(input) == 0:
            return ""
        output = TokenParser(self.ordertype).reorder(input)
        lattice = escape(output) @ self.verbalizer
        return shortestpath(lattice, nshortest=1, unique=True).string()

    def normalize(self, input, nbest=1):
        if nbest == 1:
            return self.verbalize(self.tag(input))
        return [self.verbalize(tagged) for tagged in self.tag(input, nbest)]
