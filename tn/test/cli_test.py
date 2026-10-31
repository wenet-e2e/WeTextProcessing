import inspect
import io
from types import SimpleNamespace

import pytest

import tn.cli as cli
from tn.cli import create_parser, create_processor, parse_args, run


class FakeProcessor:

    def __init__(self):
        self.tag_calls = []
        self.verbalize_calls = []

    def tag(self, text):
        self.tag_calls.append(text)
        return "<{}>".format(text)

    def verbalize(self, tagged):
        self.verbalize_calls.append(tagged)
        return "[{}]".format(tagged)


@pytest.mark.parametrize(
    "direction,language,module_name,class_name,extra_args,expected",
    [
        (
            "tn",
            "zh",
            "tn.chinese.normalizer",
            "Normalizer",
            ["--no-remove-erhua", "--tag-oov"],
            {
                "remove_erhua": False,
                "tag_oov": True
            },
        ),
        (
            "tn",
            "ja",
            "tn.japanese.normalizer",
            "Normalizer",
            ["--transliterate", "--full-to-half", "False"],
            {
                "transliterate": True,
                "full_to_half": False
            },
        ),
        ("tn", "en", "tn.english.normalizer", "Normalizer", [], {}),
        ("tn", "ko", "tn.korean.normalizer", "Normalizer", [], {}),
        ("tn", "de", "tn.german.normalizer", "Normalizer", ["--no-deterministic"], {"deterministic": False}),
        ("tn", "hy", "tn.armenian.normalizer", "Normalizer", [], {}),
        ("tn", "ar", "tn.arabic.normalizer", "Normalizer", [], {}),
        ("tn", "fr", "tn.french.normalizer", "Normalizer", [], {}),
        ("tn", "pt", "tn.portuguese.normalizer", "Normalizer", [], {}),
        ("tn", "vi", "tn.vietnamese.normalizer", "Normalizer", [], {}),
        ("tn", "ru", "tn.russian.normalizer", "Normalizer", [], {}),
        ("tn", "sv", "tn.swedish.normalizer", "Normalizer", ["--no-deterministic"], {"deterministic": False}),
        ("tn", "es", "tn.spanish.normalizer", "Normalizer", ["--no-deterministic"], {"deterministic": False}),
        ("tn", "hi", "tn.hindi.normalizer", "Normalizer", [], {}),
        ("tn", "it", "tn.italian.normalizer", "Normalizer", [], {}),
        (
            "itn",
            "zh",
            "itn.chinese.inverse_normalizer",
            "InverseNormalizer",
            ["--enable-0-to-9", "--no-enable-million"],
            {
                "enable_0_to_9": True,
                "enable_million": False
            },
        ),
        (
            "itn",
            "ja",
            "itn.japanese.inverse_normalizer",
            "InverseNormalizer",
            ["--full-to-half", "True", "--enable-standalone-number", "False"],
            {
                "full_to_half": True,
                "enable_standalone_number": False
            },
        ),
        ("itn", "en", "itn.english.inverse_normalizer", "InverseNormalizer", [], {}),
        ("itn", "ko", "itn.korean.inverse_normalizer", "InverseNormalizer", [], {}),
        ("itn", "de", "itn.german.inverse_normalizer", "InverseNormalizer", [], {}),
        ("itn", "hy", "itn.armenian.inverse_normalizer", "InverseNormalizer", [], {}),
        ("itn", "ar", "itn.arabic.inverse_normalizer", "InverseNormalizer", [], {}),
        ("itn", "fr", "itn.french.inverse_normalizer", "InverseNormalizer", [], {}),
        ("itn", "pt", "itn.portuguese.inverse_normalizer", "InverseNormalizer", [], {}),
        ("itn", "vi", "itn.vietnamese.inverse_normalizer", "InverseNormalizer", [], {}),
        ("itn", "ru", "itn.russian.inverse_normalizer", "InverseNormalizer", [], {}),
        ("itn", "sv", "itn.swedish.inverse_normalizer", "InverseNormalizer", [], {}),
        ("itn", "es", "itn.spanish.inverse_normalizer", "InverseNormalizer", ["--cased"], {"cased": True}),
        ("itn", "hi", "itn.hindi.inverse_normalizer", "InverseNormalizer", [], {}),
        ("itn", "it", "itn.italian.inverse_normalizer", "InverseNormalizer", [], {}),
    ],
)
def test_create_processor_uses_language_capabilities(
    monkeypatch,
    direction,
    language,
    module_name,
    class_name,
    extra_args,
    expected,
):
    captured = {}

    class FakeNormalizer:

        def __init__(self, **kwargs):
            captured.update(kwargs)

    module = __import__(module_name, fromlist=[class_name])
    monkeypatch.setattr(module, class_name, FakeNormalizer)
    args = parse_args(direction, ["--language", language, "--cache-dir", "/tmp/cache"] + extra_args)

    create_processor(direction, args)

    assert captured["cache_dir"] == "/tmp/cache"
    assert captured["overwrite_cache"] is False
    for name, value in expected.items():
        assert captured[name] is value


@pytest.mark.parametrize("direction", ["tn", "itn"])
def test_registered_pipeline_drives_language_options_and_factory(monkeypatch, direction):
    captured = {}
    imports = []

    class CustomNormalizer:

        def __init__(self, **kwargs):
            captured.update(kwargs)

    option = SimpleNamespace(name="custom_option", default=True, help="custom language option")
    pipeline = SimpleNamespace(module="custom.normalizer", class_name="CustomNormalizer", options=(option,))
    pipelines = {"tn": {"zh": pipeline}, "itn": {"zh": pipeline}}
    pipelines[direction]["xx"] = pipeline
    monkeypatch.setattr(cli, "PIPELINES", pipelines, raising=False)

    def import_module(name):
        imports.append(name)
        return SimpleNamespace(CustomNormalizer=CustomNormalizer)

    monkeypatch.setattr(cli, "import_module", import_module, raising=False)
    parser = create_parser(direction, "xx")
    assert "--custom-option" in parser.format_help()
    assert "custom language option" in parser.format_help()
    assert imports == []

    defaults = parse_args(direction, ["--language", "xx"])
    assert defaults.custom_option is True
    args = parse_args(direction, ["--language", "xx", "--no-custom-option", "--overwrite-cache"])
    processor = create_processor(direction, args)

    assert isinstance(processor, CustomNormalizer)
    assert captured == {"cache_dir": None, "overwrite_cache": True, "custom_option": False}
    assert imports == ["custom.normalizer"]
    other_direction = "itn" if direction == "tn" else "tn"
    with pytest.raises(SystemExit):
        parse_args(other_direction, ["--language", "xx"])


@pytest.mark.parametrize("direction, language", [
    (direction, language) for direction, languages in cli.PIPELINES.items() for language in languages
])
def test_cli_defaults_and_forwarding_match_processor_signature(monkeypatch, direction, language):
    args = parse_args(direction, ["--language", language])
    spec = cli.PIPELINES[direction][language]
    module = __import__(spec.module, fromlist=[spec.class_name])
    class_name = spec.class_name
    parameters = inspect.signature(getattr(module, class_name)).parameters
    expected = {name: parameter.default for name, parameter in parameters.items()}
    assert {name: getattr(args, name) for name in expected} == expected
    captured = {}

    class FakeNormalizer:

        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(module, class_name, FakeNormalizer)
    create_processor(direction, args)
    assert captured == expected


@pytest.mark.parametrize(
    "direction,language,present,absent",
    [
        ("tn", "zh", "--remove-erhua", "--transliterate"),
        ("tn", "en", "--cache-dir", "--remove-puncts"),
        ("tn", "ja", "--transliterate", "--remove-erhua"),
        ("tn", "ko", "--cache-dir", "--full-to-half"),
        ("tn", "de", "--deterministic", "--remove-erhua"),
        ("itn", "zh", "--remove-interjections", "--full-to-half"),
        ("itn", "en", "--cache-dir", "--enable-million"),
        ("itn", "ja", "--full-to-half", "--remove-interjections"),
        ("itn", "ko", "--cache-dir", "--enable-million"),
        ("itn", "de", "--cache-dir", "--enable-million"),
        ("tn", "hy", "--cache-dir", "--deterministic"),
        ("itn", "hy", "--cache-dir", "--enable-million"),
        ("tn", "ar", "--cache-dir", "--deterministic"),
        ("itn", "ar", "--cache-dir", "--enable-million"),
        ("tn", "fr", "--cache-dir", "--deterministic"),
        ("itn", "fr", "--cache-dir", "--enable-million"),
        ("tn", "pt", "--cache-dir", "--deterministic"),
        ("itn", "pt", "--cache-dir", "--enable-million"),
        ("tn", "vi", "--cache-dir", "--deterministic"),
        ("itn", "vi", "--cache-dir", "--enable-million"),
        ("tn", "ru", "--cache-dir", "--deterministic"),
        ("itn", "ru", "--cache-dir", "--enable-million"),
        ("tn", "sv", "--deterministic", "--remove-erhua"),
        ("itn", "sv", "--cache-dir", "--enable-million"),
        ("tn", "es", "--deterministic", "--remove-erhua"),
        ("itn", "es", "--cased", "--enable-million"),
        ("tn", "hi", "--cache-dir", "--deterministic"),
        ("itn", "hi", "--cache-dir", "--enable-million"),
        ("tn", "it", "--cache-dir", "--deterministic"),
        ("itn", "it", "--cache-dir", "--enable-million"),
    ],
)
def test_help_only_exposes_selected_pipeline_options(direction, language, present, absent):
    help_text = create_parser(direction, language).format_help()
    assert present in help_text
    assert absent not in help_text


def test_boolean_options_support_switches_and_legacy_values():
    switched = parse_args("tn", ["--remove-erhua", "--tag-oov", "--no-full-to-half"])
    legacy = parse_args("tn", ["--remove_erhua", "False", "--tag_oov", "True"])

    assert switched.remove_erhua is True
    assert switched.tag_oov is True
    assert switched.full_to_half is False
    assert legacy.remove_erhua is False
    assert legacy.tag_oov is True


def test_invalid_boolean_is_rejected():
    with pytest.raises(SystemExit):
        parse_args("tn", ["--remove-erhua", "yes"])


@pytest.mark.parametrize(
    "direction,language,option",
    [
        ("tn", "en", "--remove-puncts"),
        ("tn", "zh", "--transliterate"),
        ("itn", "en", "--enable-million"),
        ("itn", "ja", "--remove-interjections"),
    ],
)
def test_unsupported_language_option_is_rejected(direction, language, option):
    with pytest.raises(SystemExit):
        parse_args(direction, ["--language", language, option])


def test_text_is_tagged_once_and_verbalized_from_that_tag():
    processor = FakeProcessor()
    stdout = io.StringIO()

    run("tn", ["--text", "  12  "], stdout=stdout, processor_factory=lambda _direction, _args: processor)

    assert processor.tag_calls == ["  12  "]
    assert processor.verbalize_calls == ["<  12  >"]
    assert stdout.getvalue() == "<  12  >\n[<  12  >]\n"


def test_stdin_removes_only_line_endings():
    processor = FakeProcessor()
    stdout = io.StringIO()

    run(
        "itn",
        [],
        stdin=io.StringIO("  one  \n two \r\n"),
        stdout=stdout,
        processor_factory=lambda _direction, _args: processor,
    )

    assert processor.tag_calls == ["  one  ", " two "]


def test_file_input_preserves_spaces(tmp_path):
    input_path = tmp_path / "input.txt"
    input_path.write_text("  first  \n second ", encoding="utf-8")
    processor = FakeProcessor()

    run(
        "tn",
        ["--file", str(input_path)],
        stdout=io.StringIO(),
        processor_factory=lambda _direction, _args: processor,
    )

    assert processor.tag_calls == ["  first  ", " second "]


def test_text_and_file_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        parse_args("tn", ["--text", "a", "--file", "input.txt"])
