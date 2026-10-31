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
"""Shared command-line interface for text and inverse text normalization."""

import argparse
import sys
from dataclasses import dataclass
from importlib import import_module
from typing import Tuple


@dataclass(frozen=True)
class BooleanOption:
    name: str
    default: bool
    help: str


@dataclass(frozen=True)
class PipelineSpec:
    module: str
    class_name: str
    options: Tuple[BooleanOption, ...] = ()


PIPELINES = {
    "tn": {
        "zh": PipelineSpec(
            "tn.chinese.normalizer",
            "Normalizer",
            (
                BooleanOption("remove_interjections", True, 'remove interjections such as "啊"'),
                BooleanOption("remove_erhua", True, 'remove "儿" where the whitelist permits'),
                BooleanOption("traditional_to_simple", True, 'convert traditional characters, e.g. "喆" to "哲"'),
                BooleanOption("remove_puncts", False, "remove punctuation"),
                BooleanOption("full_to_half", True, 'convert full-width characters, e.g. "Ａ" to "A"'),
                BooleanOption("tag_oov", False, 'tag out-of-vocabulary characters with "OOV"'),
            ),
        ),
        "en": PipelineSpec("tn.english.normalizer", "Normalizer"),
        "ko": PipelineSpec("tn.korean.normalizer", "Normalizer"),
        "hy": PipelineSpec("tn.armenian.normalizer", "Normalizer"),
        "ar": PipelineSpec("tn.arabic.normalizer", "Normalizer"),
        "fr": PipelineSpec("tn.french.normalizer", "Normalizer"),
        "pt": PipelineSpec("tn.portuguese.normalizer", "Normalizer"),
        "vi": PipelineSpec("tn.vietnamese.normalizer", "Normalizer"),
        "ru": PipelineSpec("tn.russian.normalizer", "Normalizer"),
        "de": PipelineSpec(
            "tn.german.normalizer", "Normalizer",
            (BooleanOption("deterministic", True, "use deterministic German whitelist readings"),),
        ),
        "sv": PipelineSpec(
            "tn.swedish.normalizer", "Normalizer",
            (BooleanOption("deterministic", True, "use deterministic Swedish readings"),),
        ),
        "es": PipelineSpec(
            "tn.spanish.normalizer", "Normalizer",
            (BooleanOption("deterministic", True, "use deterministic Spanish readings"),),
        ),
        "hi": PipelineSpec("tn.hindi.normalizer", "Normalizer"),
        "ja": PipelineSpec(
            "tn.japanese.normalizer",
            "Normalizer",
            (
                BooleanOption("transliterate", False, "enable Japanese transliteration"),
                BooleanOption("remove_interjections", False, "remove interjections"),
                BooleanOption("remove_puncts", False, "remove punctuation"),
                BooleanOption("full_to_half", True, "convert full-width characters to half-width"),
                BooleanOption("tag_oov", False, 'tag out-of-vocabulary characters with "OOV"'),
            ),
        ),
    },
    "itn": {
        "zh": PipelineSpec(
            "itn.chinese.inverse_normalizer",
            "InverseNormalizer",
            (
                BooleanOption("enable_standalone_number", True, "convert standalone numbers"),
                BooleanOption("enable_0_to_9", False, "convert standalone digits from zero to nine"),
                BooleanOption("enable_million", False, "write million-scale values entirely in digits"),
                BooleanOption("remove_interjections", True, "remove interjections"),
            ),
        ),
        "en": PipelineSpec("itn.english.inverse_normalizer", "InverseNormalizer"),
        "ko": PipelineSpec("itn.korean.inverse_normalizer", "InverseNormalizer"),
        "de": PipelineSpec("itn.german.inverse_normalizer", "InverseNormalizer"),
        "hy": PipelineSpec("itn.armenian.inverse_normalizer", "InverseNormalizer"),
        "ar": PipelineSpec("itn.arabic.inverse_normalizer", "InverseNormalizer"),
        "fr": PipelineSpec("itn.french.inverse_normalizer", "InverseNormalizer"),
        "pt": PipelineSpec("itn.portuguese.inverse_normalizer", "InverseNormalizer"),
        "vi": PipelineSpec("itn.vietnamese.inverse_normalizer", "InverseNormalizer"),
        "ru": PipelineSpec("itn.russian.inverse_normalizer", "InverseNormalizer"),
        "sv": PipelineSpec("itn.swedish.inverse_normalizer", "InverseNormalizer"),
        "es": PipelineSpec(
            "itn.spanish.inverse_normalizer", "InverseNormalizer",
            (BooleanOption("cased", False, 'also accept capitalized spoken words, e.g. "Doce Dólares"'),),
        ),
        "hi": PipelineSpec("itn.hindi.inverse_normalizer", "InverseNormalizer"),
        "es_en": PipelineSpec(
            "itn.spanish_english.inverse_normalizer", "InverseNormalizer",
            (BooleanOption("cased", False, "also accept capitalized Spanish or English words"),),
        ),
        "ja": PipelineSpec(
            "itn.japanese.inverse_normalizer",
            "InverseNormalizer",
            (
                BooleanOption("enable_standalone_number", True, "convert standalone numbers"),
                BooleanOption("enable_0_to_9", False, "convert standalone digits from zero to nine"),
                BooleanOption("enable_million", False, "write million-scale values entirely in digits"),
                BooleanOption("full_to_half", False, "convert full-width characters to half-width"),
            ),
        ),
    },
}


def _pipelines(direction):
    if direction not in PIPELINES:
        raise ValueError("direction must be 'tn' or 'itn'")
    return PIPELINES[direction]


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise argparse.ArgumentTypeError("expected True or False")


def _add_boolean(parser, name, default, help):
    """Adds a Python 3.7-compatible boolean optional argument.

    ``--flag [True|False]`` keeps the historical command line working, while
    ``--flag`` and ``--no-flag`` provide conventional boolean switches.
    """

    hyphenated = name.replace("_", "-")
    parser.add_argument(
        "--" + name,
        "--" + hyphenated,
        dest=name,
        nargs="?",
        const=True,
        default=default,
        type=_parse_bool,
        metavar="BOOL",
        help=help,
    )
    parser.add_argument(
        "--no-" + name,
        "--no-" + hyphenated,
        dest=name,
        action="store_false",
        help=argparse.SUPPRESS,
    )


def _add_common_arguments(parser, languages):
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--text", help="input string; read stdin when omitted")
    inputs.add_argument("--file", help="input file path; read stdin when omitted")
    parser.add_argument("--cache_dir", "--cache-dir", default=None, help="FST cache root")
    parser.add_argument(
        "--overwrite_cache",
        "--overwrite-cache",
        action="store_true",
        help="rebuild the selected FST cache bundle",
    )
    parser.add_argument("--language", choices=languages, default="zh", help="input language (default: zh)")


def create_parser(direction, language="zh"):
    """Builds the parser for one direction/language capability set."""

    pipelines = _pipelines(direction)
    if language not in pipelines:
        raise ValueError("unsupported language: {}".format(language))
    label = "text normalization" if direction == "tn" else "inverse text normalization"
    parser = argparse.ArgumentParser(description="WeTextProcessing {}".format(label))
    _add_common_arguments(parser, tuple(pipelines))
    for option in pipelines[language].options:
        _add_boolean(parser, option.name, option.default, option.help)
    return parser


def _selected_language(direction, argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--language", choices=tuple(_pipelines(direction)), default="zh")
    args, _ = parser.parse_known_args(argv)
    return args.language


def parse_args(direction, argv=None):
    argv = sys.argv[1:] if argv is None else argv
    language = _selected_language(direction, argv)
    return create_parser(direction, language).parse_args(argv)


def create_processor(direction, args):
    pipeline = PIPELINES[direction][args.language]
    kwargs = {
        "cache_dir": args.cache_dir,
        "overwrite_cache": args.overwrite_cache,
    }
    kwargs.update({option.name: getattr(args, option.name) for option in pipeline.options})
    processor_class = getattr(import_module(pipeline.module), pipeline.class_name)
    return processor_class(**kwargs)


def _without_line_ending(line):
    if line.endswith("\n"):
        line = line[:-1]
        if line.endswith("\r"):
            line = line[:-1]
    elif line.endswith("\r"):
        line = line[:-1]
    return line


def _process_line(processor, text, stdout):
    tagged = processor.tag(text)
    print(tagged, file=stdout)
    print(processor.verbalize(tagged), file=stdout)


def run(direction, argv=None, stdin=None, stdout=None, processor_factory=create_processor):
    """Runs a CLI entry point and returns the constructed processor."""

    args = parse_args(direction, argv)
    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
    processor = processor_factory(direction, args)

    if args.text is not None:
        _process_line(processor, args.text, stdout)
        return processor

    if args.file is not None:
        try:
            with open(args.file, encoding="utf-8") as input_file:
                for line in input_file:
                    _process_line(processor, _without_line_ending(line), stdout)
        except OSError as error:
            create_parser(direction, args.language).error(str(error))
        return processor

    for line in stdin:
        _process_line(processor, _without_line_ending(line), stdout)
    return processor
