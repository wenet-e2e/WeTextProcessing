import json
import multiprocessing
import os
import time
from pathlib import Path

import pytest
from pynini import cross

import tn.cache as cache
from tn.cache import CacheBundle, CacheLockTimeout, CachePathError
from tn.chinese.test.processor_test import CountingProcessor
from tn.processor import Processor


class MultiprocessCountingProcessor(Processor):

    def __init__(self, cache_dir):
        super().__init__("multiprocess_counting")
        self.build_fst("zh_tn", cache_dir, False, {})

    def build_tagger_and_verbalizer(self):
        marker = os.environ["WETEXT_TEST_BUILD_MARKER"]
        descriptor = os.open(marker, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        try:
            os.write(descriptor, b"build\n")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        time.sleep(0.1)
        tagged = 'multiprocess_counting { value: "output" }'
        self.tagger = cross("input", tagged)
        self.verbalizer = cross(tagged, "output")


def _bundle(cache_root):
    return CacheBundle(cache_root, "zh_tn", "tn", {}, "test-source")


def _multiprocess_build(cache_dir):
    return MultiprocessCountingProcessor(cache_dir).normalize("input")


def _hold_lock(cache_root, ready, release):
    with _bundle(cache_root).lock():
        ready.set()
        release.wait(10)


def _crash_with_lock(cache_root, ready):
    with _bundle(cache_root).lock():
        ready.set()
        os._exit(0)


def test_four_processes_build_one_bundle(monkeypatch, tmp_path):
    marker = tmp_path / "builds.txt"
    cache_root = tmp_path / "cache"
    monkeypatch.setenv("WETEXT_TEST_BUILD_MARKER", os.fspath(marker))
    context = multiprocessing.get_context("spawn")

    with context.Pool(processes=4) as pool:
        outputs = pool.map(_multiprocess_build, [os.fspath(cache_root)] * 4)

    assert outputs == ["output"] * 4
    assert marker.read_text(encoding="utf-8").splitlines() == ["build"]
    assert len(list(cache_root.glob("**/manifest.json"))) == 1


def test_persistent_anchor_is_reused(tmp_path):
    bundle = _bundle(tmp_path)

    with bundle.lock():
        anchor_stat = os.lstat(bundle.lock_path)
    assert bundle.lock_path.is_file()
    with bundle.lock():
        assert os.lstat(bundle.lock_path).st_ino == anchor_stat.st_ino
    assert bundle.lock_path.is_file()


def test_active_owner_timeout_reports_bounded_payload(tmp_path):
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    process = context.Process(target=_hold_lock, args=(os.fspath(tmp_path), ready, release))
    process.start()
    assert ready.wait(5)
    try:
        with pytest.raises(CacheLockTimeout, match="hostname"):
            with _bundle(tmp_path).lock(timeout=0.02):
                pass
    finally:
        release.set()
        process.join(5)
    assert process.exitcode == 0


def test_owner_crash_releases_os_lock_immediately(tmp_path):
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    process = context.Process(target=_crash_with_lock, args=(os.fspath(tmp_path), ready))
    process.start()
    assert ready.wait(5)
    process.join(5)
    assert process.exitcode == 0

    with _bundle(tmp_path).lock(timeout=0.5):
        pass


@pytest.mark.parametrize(
    "payload,expected",
    [
        (b"{", "unreadable owner payload"),
        (None, "owner payload too large"),
    ],
)
def test_partial_or_giant_owner_payload_is_bounded_diagnostic(tmp_path, payload, expected):
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    process = context.Process(target=_hold_lock, args=(os.fspath(tmp_path), ready, release))
    process.start()
    assert ready.wait(5)
    bundle = _bundle(tmp_path)
    descriptor = os.open(bundle.lock_path, os.O_RDWR)
    try:
        if payload is None:
            os.lseek(descriptor, cache._OWNER_PAYLOAD_MAX_BYTES + 1, os.SEEK_SET)
            os.write(descriptor, b"x")
        else:
            payload_size = os.fstat(descriptor).st_size - 1
            os.lseek(descriptor, 1, os.SEEK_SET)
            os.write(descriptor, payload + b" " * (payload_size - len(payload)))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        with pytest.raises(CacheLockTimeout, match=expected):
            with bundle.lock(timeout=0.02):
                pass
    finally:
        release.set()
        process.join(5)
    assert process.exitcode == 0


def test_lock_timeout_can_be_configured_by_environment(monkeypatch, tmp_path):
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    process = context.Process(target=_hold_lock, args=(os.fspath(tmp_path), ready, release))
    process.start()
    assert ready.wait(5)
    monkeypatch.setenv(cache.LOCK_TIMEOUT_ENV, "0")
    try:
        started = time.monotonic()
        with pytest.raises(CacheLockTimeout):
            with _bundle(tmp_path).lock():
                pass
        assert time.monotonic() - started < 0.5
    finally:
        release.set()
        process.join(5)


@pytest.mark.parametrize("value", ["nan", "inf", "-1", "not-a-number"])
def test_invalid_lock_timeout_environment_is_rejected(monkeypatch, tmp_path, value):
    monkeypatch.setenv(cache.LOCK_TIMEOUT_ENV, value)

    with pytest.raises(ValueError, match="non-negative number"):
        with _bundle(tmp_path).lock():
            pass


def test_owner_payload_write_handles_short_writes(monkeypatch, tmp_path):
    real_write = cache.os.write
    calls = []

    def short_write(descriptor, data):
        calls.append(len(data))
        return real_write(descriptor, data[:1])

    monkeypatch.setattr(cache.os, "write", short_write)
    with _bundle(tmp_path).lock():
        pass

    assert len(calls) > 1


def test_owner_payload_failure_releases_lock_but_keeps_anchor(monkeypatch, tmp_path):
    bundle = _bundle(tmp_path)
    original = CacheBundle._write_owner_payload

    def fail_payload(descriptor, payload):
        raise OSError("injected owner payload failure")

    monkeypatch.setattr(CacheBundle, "_write_owner_payload", staticmethod(fail_payload))
    with pytest.raises(OSError, match="injected"):
        with bundle.lock():
            pass
    assert bundle.lock_path.is_file()

    monkeypatch.setattr(CacheBundle, "_write_owner_payload", staticmethod(original))
    with bundle.lock(timeout=0.1):
        pass


class _FakeMsvcrt:
    LK_NBLCK = 1
    LK_UNLCK = 2

    def __init__(self):
        self.calls = []

    def locking(self, descriptor, mode, length):
        self.calls.append((descriptor, mode, length))


def test_windows_msvcrt_lock_and_unlock_keep_full_descriptor_value():
    descriptor = 1 << 40
    module = _FakeMsvcrt()
    seeks = []

    cache._windows_try_lock(descriptor, msvcrt_module=module, seek=lambda *args: seeks.append(args))
    cache._windows_unlock(descriptor, msvcrt_module=module, seek=lambda *args: seeks.append(args))

    assert module.calls == [
        (descriptor, module.LK_NBLCK, 1),
        (descriptor, module.LK_UNLCK, 1),
    ]
    assert seeks == [
        (descriptor, 0, os.SEEK_SET),
        (descriptor, 0, os.SEEK_SET),
    ]


@pytest.mark.parametrize("prefix", [".", "..", "../escape", "a/b", "a\\b"])
def test_prefix_must_be_one_safe_component(tmp_path, prefix):
    with pytest.raises(ValueError, match="one safe path component"):
        CacheBundle(tmp_path, prefix, "tn", {}, "source")


def test_symlinked_cache_child_is_rejected(tmp_path):
    if not hasattr(os, "symlink"):
        pytest.skip("symlinks are unavailable")
    cache_root = tmp_path / "cache"
    outside = tmp_path / "outside"
    cache_root.mkdir()
    outside.mkdir()
    (cache_root / "zh_tn").symlink_to(outside, target_is_directory=True)

    with pytest.raises((CachePathError, OSError)):
        with CacheBundle(cache_root, "zh_tn", "tn", {}, "source").lock():
            pass


def test_cache_hit_rejects_symlinked_prefix_ancestor(tmp_path):
    if not hasattr(os, "symlink"):
        pytest.skip("symlinks are unavailable")
    CountingProcessor(tmp_path)
    prefix = tmp_path / "zh_tn"
    outside = tmp_path / "outside-prefix"
    os.replace(prefix, outside)
    prefix.symlink_to(outside, target_is_directory=True)

    with pytest.raises((CachePathError, OSError)):
        CountingProcessor(tmp_path)


def test_symlinked_lock_anchor_is_rejected(tmp_path):
    if not hasattr(os, "symlink"):
        pytest.skip("symlinks are unavailable")
    bundle = _bundle(tmp_path)
    bundle._ensure_parent()
    bundle.lock_path.symlink_to(tmp_path / "missing-lock")

    with pytest.raises((CachePathError, OSError)):
        with bundle.lock(timeout=0):
            pass


def test_symlinked_manifest_is_rejected(tmp_path):
    if not hasattr(os, "symlink"):
        pytest.skip("symlinks are unavailable")
    CountingProcessor(tmp_path)
    manifest_path = next(tmp_path.glob("**/manifest.json"))
    target = tmp_path / "outside.json"
    target.write_bytes(manifest_path.read_bytes())
    manifest_path.unlink()
    manifest_path.symlink_to(target)

    with pytest.raises((CachePathError, OSError)):
        CountingProcessor(tmp_path)


def test_symlinked_fst_is_rejected_before_native_read(tmp_path):
    if not hasattr(os, "symlink"):
        pytest.skip("symlinks are unavailable")
    CountingProcessor(tmp_path)
    manifest_path = next(tmp_path.glob("**/manifest.json"))
    tagger_path = manifest_path.parent / "tagger.fst"
    target = tmp_path / "outside.fst"
    target.write_bytes(tagger_path.read_bytes())
    tagger_path.unlink()
    tagger_path.symlink_to(target)

    with pytest.raises((CachePathError, OSError)):
        CountingProcessor(tmp_path)


def test_broken_bundle_symlink_is_rejected(tmp_path):
    if not hasattr(os, "symlink"):
        pytest.skip("symlinks are unavailable")
    bundle = _bundle(tmp_path)
    bundle._ensure_parent()
    bundle.path.symlink_to(tmp_path / "missing", target_is_directory=True)

    with pytest.raises((CachePathError, OSError)):
        bundle.load()


@pytest.mark.parametrize("manifest_value", [[], None])
def test_non_dict_manifest_is_treated_as_corrupt_and_rebuilt(tmp_path, manifest_value):
    CountingProcessor.builds = 0
    CountingProcessor(tmp_path)
    manifest_path = next(tmp_path.glob("**/manifest.json"))
    manifest_path.write_text(json.dumps(manifest_value), encoding="utf-8")

    assert CountingProcessor(tmp_path).normalize("input") == "output"
    assert CountingProcessor.builds == 2


def test_manifest_size_limit_is_checked_before_loading(tmp_path):
    CountingProcessor.builds = 0
    CountingProcessor(tmp_path)
    manifest_path = next(tmp_path.glob("**/manifest.json"))
    with open(manifest_path, "wb") as manifest_file:
        manifest_file.truncate(cache._MANIFEST_MAX_BYTES + 1)

    assert CountingProcessor(tmp_path).normalize("input") == "output"
    assert CountingProcessor.builds == 2


def test_read_from_string_accepts_single_bytearray_buffer():
    serialized = cross("input", "output").write_to_string()

    assert cache.Fst.read_from_string(bytearray(serialized)).num_states() > 0


def test_previous_complete_bundle_is_recovered_after_crash(tmp_path):
    if os.name == "nt":
        pytest.skip("Windows intentionally leaves residuals untouched")
    CountingProcessor.builds = 0
    CountingProcessor(tmp_path)
    fingerprint = CountingProcessor._source_fingerprint("zh_tn")
    bundle = CacheBundle(tmp_path, "zh_tn", "tn", {}, fingerprint)
    previous = bundle.parent / ".{}.previous-crash".format(bundle.bundle_digest)
    os.replace(bundle.path, previous)

    processor = CountingProcessor(tmp_path)

    assert processor.normalize("input") == "output"
    assert CountingProcessor.builds == 1
    assert bundle.path.exists()
    assert not previous.exists()


def test_residual_cleanup_never_follows_nested_symlink(tmp_path):
    if not hasattr(os, "symlink") or os.name == "nt":
        pytest.skip("POSIX cleanup behavior")
    bundle = _bundle(tmp_path)
    bundle._ensure_parent()
    outside = tmp_path / "outside"
    outside.mkdir()
    protected = outside / "protected.txt"
    protected.write_text("keep", encoding="utf-8")
    residual = bundle.parent / ".{}.tmp-crash".format(bundle.bundle_digest)
    residual.mkdir()
    (residual / "outside-link").symlink_to(outside, target_is_directory=True)

    bundle.recover_residuals()

    assert not residual.exists()
    assert protected.read_text(encoding="utf-8") == "keep"


def test_unexpected_residual_subdirectory_is_preserved(tmp_path):
    bundle = _bundle(tmp_path)
    bundle._ensure_parent()
    residual = bundle.parent / ".{}.tmp-review".format(bundle.bundle_digest)
    unexpected = residual / "unexpected"
    unexpected.mkdir(parents=True)

    bundle.recover_residuals()

    assert unexpected.is_dir()


def test_successful_publish_is_not_failed_by_unremovable_previous(tmp_path):
    if os.name == "nt":
        pytest.skip("POSIX cleanup behavior")
    CountingProcessor.builds = 0
    CountingProcessor(tmp_path)
    manifest_path = next(tmp_path.glob("**/manifest.json"))
    (manifest_path.parent / "unexpected").mkdir()

    processor = CountingProcessor(tmp_path, overwrite_cache=True)

    assert processor.normalize("input") == "output"
    assert CountingProcessor.builds == 2
    assert list(manifest_path.parent.parent.glob("*.previous-*"))


def test_cleanup_failure_does_not_mask_publication_error(monkeypatch, tmp_path):

    def fail_manifest(path, manifest):
        (path.parent / "unexpected").mkdir()
        raise OSError("primary publication failure")

    monkeypatch.setattr(CacheBundle, "_write_manifest", staticmethod(fail_manifest))

    with pytest.raises(OSError, match="primary publication failure"):
        CountingProcessor(tmp_path)


def test_windows_cleanup_is_disabled_without_unlink(monkeypatch, tmp_path):
    residual = tmp_path / "residual"
    residual.mkdir()
    (residual / "tagger.fst").write_bytes(b"fst")
    calls = []

    def forbidden_unlink(path):
        calls.append(path)
        raise AssertionError("Windows cleanup must not unlink")

    monkeypatch.setattr(cache.os, "name", "nt")
    monkeypatch.setattr(Path, "unlink", forbidden_unlink)

    assert not cache._remove_flat_bundle(residual, best_effort=True)
    assert calls == []
    assert residual.exists()


def test_windows_replace_retries_a_reader_conflict(monkeypatch, tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.write_text("complete", encoding="utf-8")
    real_replace = cache.os.replace
    calls = {"count": 0}

    def conflict_once(first, second):
        calls["count"] += 1
        if calls["count"] == 1:
            raise PermissionError("reader holds file")
        return real_replace(first, second)

    monkeypatch.setattr(cache.os, "name", "nt")
    monkeypatch.setattr(cache.os, "replace", conflict_once)
    monkeypatch.setattr(cache.time, "sleep", lambda seconds: None)

    CacheBundle._replace_with_retry(source, destination, "test replace")

    assert destination.read_text(encoding="utf-8") == "complete"
    assert calls["count"] == 2


def test_reparse_point_is_detected():

    class ReparseStat:
        st_file_attributes = 0x400

    assert cache._is_reparse_point(ReparseStat())


def test_new_default_cache_root_is_private(monkeypatch, tmp_path):
    if os.name == "nt":
        pytest.skip("POSIX permission bits are not authoritative on Windows")
    xdg = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CACHE_HOME", os.fspath(xdg))

    CountingProcessor(None)

    assert cache.default_cache_dir().stat().st_mode & 0o777 == 0o700


def test_relative_xdg_cache_home_is_not_used(monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", "relative-cache")

    result = cache.default_cache_dir()

    assert result.is_absolute()
    assert result != Path.cwd() / "relative-cache" / "wetextprocessing"


def test_preexisting_explicit_root_permissions_are_not_changed(tmp_path):
    if os.name == "nt":
        pytest.skip("POSIX permission bits are not authoritative on Windows")
    explicit = tmp_path / "explicit"
    explicit.mkdir(mode=0o755)
    explicit.chmod(0o755)

    CountingProcessor(explicit)

    assert explicit.stat().st_mode & 0o777 == 0o755
