import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from pynini import accep, cross, escape, union
from pynini.lib.pynutil import add_weight

from tn.cache import CacheBundle, default_cache_dir, production_source_fingerprint
from tn.processor import Processor, _UniqueOutputPathStream


class CountingProcessor(Processor):

    builds = 0

    def __init__(self, cache_dir, cache_config=None, overwrite_cache=False):
        super().__init__("counting")
        self.build_fst("zh_tn", cache_dir, overwrite_cache, cache_config or {})

    def build_tagger_and_verbalizer(self):
        type(self).builds += 1
        self.tagger = cross("input", 'counting { value: "output" }')
        self.verbalizer = cross('counting { value: "output" }', "output")


class AmbiguousProcessor(Processor):

    def __init__(self, cache_dir):
        super().__init__("counting")
        self.build_fst("zh_tn", cache_dir, False, {})

    def build_tagger_and_verbalizer(self):
        self.tagger = union(
            cross("input", 'counting { value: "first" }'),
            cross("input", 'counting { value: "second" }'),
        )
        self.verbalizer = union(
            cross('counting { value: "first" }', "FIRST"),
            cross('counting { value: "second" }', "SECOND"),
        )


class RawAmbiguousProcessor(Processor):

    def __init__(self):
        super().__init__("counting")
        tagged = 'counting { value: "input" }'
        self.tagger = cross("input", tagged)
        self.verbalizer = union(
            cross(tagged, "FIRST"),
            add_weight(cross(tagged, "SECOND"), 1.0),
        )


class WeightedJointProcessor(Processor):

    def __init__(self):
        super().__init__("counting")
        first = 'counting { value: "first" }'
        second = 'counting { value: "second" }'
        self.tagger = union(
            cross("input", first),
            add_weight(cross("input", second), 1.0),
        )
        self.verbalizer = union(
            add_weight(cross(first, "FIRST_BEST"), 5.0),
            add_weight(cross(first, "FIRST_ALT"), 7.0),
            add_weight(cross(second, "SECOND_BEST"), -10.0),
            add_weight(cross(second, "SECOND_ALT"), -9.0),
        )


class JointItnProcessor(Processor):

    def __init__(self):
        super().__init__("counting", ordertype="itn")
        first = 'counting { value: "first" }'
        second = 'counting { value: "second" }'
        self.tagger = union(
            cross("input", first),
            add_weight(cross("input", second), 1.0),
        )
        self.verbalizer = union(
            cross(first, "A"),
            add_weight(cross(first, "ALT"), 1.0),
            cross(second, "B"),
        )


class WeightedFieldProcessor(Processor):

    def __init__(self):
        super().__init__("weighted", ordertype="itn")
        self.graph = union(
            add_weight(cross("input", "BEST"), 2.0),
            add_weight(cross("input", "ALT"), 5.0),
        )
        self.tagger = self.add_tokens(self.tag_field("value", self.graph))
        self.verbalizer = self.delete_tokens(self.verbalize_field("value", self.graph))


def test_cache_is_reused_only_for_matching_config(tmp_path):
    CountingProcessor.builds = 0

    CountingProcessor(tmp_path, {"option": True})
    CountingProcessor(tmp_path, {"option": True})
    assert CountingProcessor.builds == 1

    CountingProcessor(tmp_path, {"option": False})
    assert CountingProcessor.builds == 2

    CountingProcessor(tmp_path, {"option": True})
    assert CountingProcessor.builds == 2


def test_cache_is_rebuilt_when_source_fingerprint_differs(tmp_path):
    CountingProcessor.builds = 0
    CountingProcessor(tmp_path, {})

    manifest_path = next(tmp_path.glob("**/manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_fingerprint"] = "stale"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    CountingProcessor(tmp_path, {})
    assert CountingProcessor.builds == 2


def test_source_fingerprint_is_content_addressed_and_reusable(monkeypatch, tmp_path):
    current = {"fingerprint": "source-a"}
    monkeypatch.setattr(
        CountingProcessor,
        "_source_fingerprint",
        staticmethod(lambda prefix: current["fingerprint"]),
    )
    CountingProcessor.builds = 0

    CountingProcessor(tmp_path)
    CountingProcessor(tmp_path)
    current["fingerprint"] = "source-b"
    CountingProcessor(tmp_path)
    current["fingerprint"] = "source-a"
    CountingProcessor(tmp_path)

    assert CountingProcessor.builds == 2
    assert len(list(tmp_path.glob("**/manifest.json"))) == 2


def test_default_cache_uses_xdg_user_cache(monkeypatch, tmp_path):
    xdg_cache_home = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CACHE_HOME", os.fspath(xdg_cache_home))
    CountingProcessor.builds = 0

    CountingProcessor(None)

    assert default_cache_dir() == xdg_cache_home / "wetextprocessing"
    assert list(default_cache_dir().glob("**/manifest.json"))
    assert not list(tmp_path.glob("zh_tn_*.fst"))


def test_explicit_cache_dir_is_used_as_bundle_root(tmp_path):
    explicit = tmp_path / "explicit"

    CountingProcessor(explicit)

    manifests = list(explicit.glob("zh_tn/*/*/manifest.json"))
    assert len(manifests) == 1


def test_cache_can_be_disabled_without_writing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    CountingProcessor.builds = 0

    CountingProcessor(False)
    CountingProcessor(False)

    assert CountingProcessor.builds == 2
    assert list(tmp_path.iterdir()) == []


def test_production_fingerprint_uses_content_not_mtime(tmp_path):
    source = tmp_path / "tn" / "rules" / "rule.py"
    resource = tmp_path / "itn" / "english" / "data" / "number.tsv"
    test_source = tmp_path / "tn" / "chinese" / "test" / "ignored.py"
    source.parent.mkdir(parents=True)
    resource.parent.mkdir(parents=True)
    test_source.parent.mkdir(parents=True)
    source.write_text("RULE = 1\n", encoding="utf-8")
    resource.write_text("one\t1\n", encoding="utf-8")
    test_source.write_text("IGNORED = 1\n", encoding="utf-8")

    original = production_source_fingerprint(tmp_path)
    stat = source.stat()
    os.utime(source, (stat.st_atime + 10, stat.st_mtime + 10))
    assert production_source_fingerprint(tmp_path) == original

    source.write_text("RULE = 2\n", encoding="utf-8")
    source_changed = production_source_fingerprint(tmp_path)
    assert source_changed != original

    resource.write_text("two\t2\n", encoding="utf-8")
    assert production_source_fingerprint(tmp_path) != source_changed

    before_test_change = production_source_fingerprint(tmp_path)
    test_source.write_text("IGNORED = 2\n", encoding="utf-8")
    assert production_source_fingerprint(tmp_path) == before_test_change


class SlowCountingProcessor(CountingProcessor):

    builds = 0

    def build_tagger_and_verbalizer(self):
        time.sleep(0.1)
        super().build_tagger_and_verbalizer()


def test_concurrent_builders_publish_one_complete_bundle(tmp_path):
    SlowCountingProcessor.builds = 0

    with ThreadPoolExecutor(max_workers=2) as executor:
        processors = list(executor.map(lambda _: SlowCountingProcessor(tmp_path), range(2)))

    assert SlowCountingProcessor.builds == 1
    assert [processor.normalize("input") for processor in processors] == ["output", "output"]
    manifest_path = next(tmp_path.glob("**/manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(manifest["files"]) == {"tagger.fst", "verbalizer.fst"}


@pytest.mark.parametrize("failure", ["tagger", "verbalizer", "manifest"])
def test_failed_publication_preserves_previous_bundle(monkeypatch, tmp_path, failure):
    CountingProcessor.builds = 0
    CountingProcessor(tmp_path)
    original_write_fst = CacheBundle._write_fst
    original_write_manifest = CacheBundle._write_manifest
    write_count = {"fst": 0}

    def failing_write_fst(graph, path):
        write_count["fst"] += 1
        if failure == "tagger" and write_count["fst"] == 1:
            raise OSError("injected tagger failure")
        if failure == "verbalizer" and write_count["fst"] == 2:
            raise OSError("injected verbalizer failure")
        return original_write_fst(graph, path)

    def failing_write_manifest(path, manifest):
        if failure == "manifest":
            raise OSError("injected manifest failure")
        return original_write_manifest(path, manifest)

    monkeypatch.setattr(CacheBundle, "_write_fst", staticmethod(failing_write_fst))
    monkeypatch.setattr(CacheBundle, "_write_manifest", staticmethod(failing_write_manifest))
    with pytest.raises(OSError, match="injected"):
        CountingProcessor(tmp_path, overwrite_cache=True)

    monkeypatch.setattr(CacheBundle, "_write_fst", staticmethod(original_write_fst))
    monkeypatch.setattr(CacheBundle, "_write_manifest", staticmethod(original_write_manifest))
    processor = CountingProcessor(tmp_path)
    assert processor.normalize("input") == "output"
    assert CountingProcessor.builds == 2
    assert not [path for path in tmp_path.rglob("*") if ".tmp-" in path.name]
    anchors = [path for path in tmp_path.rglob("*.lock")]
    assert len(anchors) == 1
    assert anchors[0].is_file()


@pytest.mark.parametrize("damage", ["missing", "corrupt", "manifest"])
def test_incomplete_or_corrupt_bundle_is_rebuilt(tmp_path, damage):
    CountingProcessor.builds = 0
    CountingProcessor(tmp_path)
    manifest_path = next(tmp_path.glob("**/manifest.json"))
    bundle_path = manifest_path.parent
    if damage == "missing":
        (bundle_path / "tagger.fst").unlink()
    elif damage == "corrupt":
        (bundle_path / "verbalizer.fst").write_bytes(b"not an fst")
    else:
        manifest_path.write_text("{", encoding="utf-8")

    processor = CountingProcessor(tmp_path)

    assert CountingProcessor.builds == 2
    assert processor.normalize("input") == "output"
    assert json.loads(next(tmp_path.glob("**/manifest.json")).read_text(encoding="utf-8"))["cache_format"] == 2


def test_legacy_flat_cache_is_left_untouched_and_not_trusted(tmp_path):
    legacy_tagger = tmp_path / "zh_tn_tagger.fst"
    legacy_verbalizer = tmp_path / "zh_tn_verbalizer.fst"
    legacy_manifest = tmp_path / "zh_tn_cache.json"
    legacy_tagger.write_bytes(b"legacy tagger")
    legacy_verbalizer.write_bytes(b"legacy verbalizer")
    legacy_manifest.write_text('{"version": 1}', encoding="utf-8")

    CountingProcessor.builds = 0
    processor = CountingProcessor(tmp_path)

    assert CountingProcessor.builds == 1
    assert processor.normalize("input") == "output"
    assert legacy_tagger.read_bytes() == b"legacy tagger"
    assert legacy_verbalizer.read_bytes() == b"legacy verbalizer"
    assert json.loads(legacy_manifest.read_text(encoding="utf-8")) == {"version": 1}


def test_failed_initial_publication_leaves_no_readable_partial_bundle(monkeypatch, tmp_path):

    def fail_manifest(path, manifest):
        raise OSError("injected manifest failure")

    monkeypatch.setattr(CacheBundle, "_write_manifest", staticmethod(fail_manifest))
    with pytest.raises(OSError, match="injected"):
        CountingProcessor(tmp_path)

    assert not list(tmp_path.glob("**/manifest.json"))
    assert not list(tmp_path.glob("**/*.fst"))
    assert not [path for path in tmp_path.rglob("*") if ".tmp-" in path.name]
    anchors = [path for path in tmp_path.rglob("*.lock")]
    assert len(anchors) == 1
    assert anchors[0].is_file()


def test_normalize_with_mapping_uses_tagged_token_path(tmp_path):
    processor = CountingProcessor(tmp_path, {})

    result = processor.normalize_with_mapping("input")

    assert result.output_text == "output"
    assert len(result.mappings) == 1
    assert result.mappings[0].token_type == "counting"
    assert result.mappings[0].input_text == "input"
    assert result.mappings[0].output_text == "output"


def test_normalize_with_mapping_supports_nbest(tmp_path):
    processor = AmbiguousProcessor(tmp_path)

    results = processor.normalize_with_mapping("input", nbest=2)

    assert {result.output_text for result in results} == {"FIRST", "SECOND"}
    assert all(result.mappings[0].input_text == "input" for result in results)


def test_nbest_combines_raw_tagger_with_verbalizer_paths():
    processor = RawAmbiguousProcessor()

    assert processor.tag("input", nbest=2) == ['counting { value: "input" }']
    assert processor.normalize("input", nbest=2) == ["FIRST", "SECOND"]

    results = processor.normalize_with_mapping("input", nbest=2)
    assert [result.output_text for result in results] == ["FIRST", "SECOND"]
    assert [result.mappings[0].output_text for result in results] == ["FIRST", "SECOND"]


def test_nbest_combines_tagger_weight_with_per_tag_verbalizer_delta():
    processor = WeightedJointProcessor()

    assert processor.normalize("input", nbest=4) == [
        "FIRST_BEST",
        "SECOND_BEST",
        "FIRST_ALT",
        "SECOND_ALT",
    ]


def test_itn_normalize_and_mapping_share_joint_candidates():
    processor = JointItnProcessor()

    outputs = processor.normalize("input", nbest=3)
    results = processor.normalize_with_mapping("input", nbest=2)
    assert outputs == ["A", "ALT", "B"]
    assert [result.output_text for result in results] == outputs[:2]
    assert [result.mappings[0].output_text for result in results] == outputs[:2]


def test_weighted_field_restores_original_joint_path_costs():
    processor = WeightedFieldProcessor()

    candidates = processor._normalization_candidates("input", nbest=2)

    assert [candidate.output for candidate in candidates] == ["BEST", "ALT"]
    assert [candidate.tagger_weight for candidate in candidates] == pytest.approx([2.0, 2.0])
    assert [candidate.weight for candidate in candidates] == pytest.approx([2.0, 5.0])


def test_weighted_stream_projects_unique_outputs_before_expanding():
    tagged = 'counting { value: "input" }'
    duplicate_a_paths = [add_weight(cross(tagged, "A"), index / 2048.0) for index in range(2048)]
    verbalizer = union(*duplicate_a_paths, add_weight(cross(tagged, "B"), 1.0))
    stream = _UniqueOutputPathStream(accep(escape(tagged)) @ verbalizer)

    assert [stream.pop().text, stream.pop().text] == ["A", "B"]
    assert stream._limit == 2


@pytest.mark.parametrize("nbest", [0, -1, 1.5, True, None])
def test_nbest_must_be_a_positive_integer(tmp_path, nbest):
    processor = CountingProcessor(tmp_path, {})

    with pytest.raises(ValueError, match="positive integer"):
        processor.normalize("", nbest=nbest)


def test_rule_inventory_requires_a_tagger_and_verbalizer():
    from tn.processor import Processor, RuleSpec

    rule = type("Rule", (), {"tagger": cross("a", "b"), "verbalizer": cross("b", "c")})()
    verbalizer_only = [RuleSpec(rule)]
    tagger_only = [RuleSpec(rule, 1.0, verbalize=False)]

    with pytest.raises(ValueError, match="tagger"):
        Processor.tagger_union(verbalizer_only)
    with pytest.raises(ValueError, match="verbalizer"):
        Processor.verbalizer_union(tagger_only)


def test_pipeline_owned_token_schema_is_used():
    processor = Processor("schema", token_orders={"custom": ["second", "first"]})

    parser = processor.token_parser()

    assert parser.reorder('custom { first: "1" second: "2" }') == 'custom { second: "2" first: "1" }'
