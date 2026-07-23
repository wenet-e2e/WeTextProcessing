import json

from pynini import cross

from tn.processor import Processor


class CountingProcessor(Processor):

    builds = 0

    def __init__(self, cache_dir, cache_config):
        super().__init__("counting")
        self.build_fst("zh_tn", cache_dir, False, cache_config)

    def build_tagger_and_verbalizer(self):
        type(self).builds += 1
        self.tagger = cross("input", 'counting { value: "output" }')
        self.verbalizer = cross('counting { value: "output" }', "output")


def test_cache_is_reused_only_for_matching_config(tmp_path):
    CountingProcessor.builds = 0

    CountingProcessor(tmp_path, {"option": True})
    CountingProcessor(tmp_path, {"option": True})
    assert CountingProcessor.builds == 1

    CountingProcessor(tmp_path, {"option": False})
    assert CountingProcessor.builds == 2


def test_cache_is_rebuilt_when_source_fingerprint_differs(tmp_path):
    CountingProcessor.builds = 0
    CountingProcessor(tmp_path, {})

    manifest_path = tmp_path / "zh_tn_cache.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_fingerprint"] = "stale"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    CountingProcessor(tmp_path, {})
    assert CountingProcessor.builds == 2
