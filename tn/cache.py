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
"""Persistent, content-addressed storage for compiled Pynini graphs."""

import hashlib
import errno
import json
import math
import os
import platform
import re
import stat
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import pynini
from pynini import Fst

CACHE_FORMAT_VERSION = 2
BUILDER_ABI_VERSION = 1
DEFAULT_LOCK_TIMEOUT_SECONDS = 10 * 60
LOCK_TIMEOUT_ENV = "WETEXTPROCESSING_CACHE_LOCK_TIMEOUT"
_OWNER_PAYLOAD_MAX_BYTES = 64 * 1024
_CACHE_APPLICATION = "wetextprocessing"
_PRODUCTION_SUFFIXES = frozenset((".py", ".tsv", ".far"))
_SAFE_PREFIX = re.compile(r"^[A-Za-z0-9_.-]+$")
_BUNDLE_FILENAMES = ("tagger.fst", "verbalizer.fst", "manifest.json")
_MANIFEST_MAX_BYTES = 1024 * 1024
_FST_MAX_BYTES = min(sys.maxsize, 2 * 1024 * 1024 * 1024)


class CacheError(RuntimeError):
    pass


class CachePathError(CacheError):
    pass


class CacheIntegrityError(CacheError, ValueError):
    pass


class CacheLockTimeout(CacheError):

    def __init__(self, path, timeout, owner):
        message = "timed out after {:.3f}s waiting for cache lock {} (owner: {})".format(
            timeout,
            path,
            owner,
        )
        super().__init__(message)
        self.path = path
        self.timeout = timeout
        self.owner = owner


def default_cache_dir():
    """Returns a stable per-user cache directory without creating it."""

    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    xdg_path = Path(xdg_cache_home).expanduser() if xdg_cache_home else None
    if xdg_path is not None and xdg_path.is_absolute():
        base = xdg_path
    elif os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        local_path = Path(local_app_data).expanduser() if local_app_data else None
        base = local_path if local_path is not None and local_path.is_absolute() else Path.home() / "AppData" / "Local"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path.home() / ".cache"
    return base / _CACHE_APPLICATION


def _canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _hash_bytes(value):
    return hashlib.sha256(value).hexdigest()


def production_source_fingerprint(project_root=None):
    """Hashes graph-producing Python and data resources by content."""

    project_root = Path(__file__).resolve().parent.parent if project_root is None else Path(project_root)
    sources = []
    for package_name in ("tn", "itn"):
        package_root = project_root / package_name
        for path in package_root.rglob("*"):
            if not path.is_file() or path.suffix not in _PRODUCTION_SUFFIXES:
                continue
            relative_parts = path.relative_to(package_root).parts
            if "test" in relative_parts or "__pycache__" in relative_parts:
                continue
            if path.name == "_version.py":
                continue
            sources.append(path)

    digest = hashlib.sha256()
    for source in sorted(sources):
        relative_path = source.relative_to(project_root).as_posix().encode("utf-8")
        digest.update(len(relative_path).to_bytes(8, "big"))
        digest.update(relative_path)
        with open(source, "rb") as source_file:
            for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _builder_identity():
    return {
        "abi": BUILDER_ABI_VERSION,
        "byteorder": sys.byteorder,
        "cache_format": CACHE_FORMAT_VERSION,
        "implementation": platform.python_implementation(),
        "pynini": getattr(pynini, "__version__", "unknown"),
        "python": "{}.{}".format(sys.version_info[0], sys.version_info[1]),
    }


def _fsync_directory(path):
    try:
        descriptor = os.open(os.fspath(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _configured_lock_timeout():
    value = os.environ.get(LOCK_TIMEOUT_ENV)
    if value is None:
        return float(DEFAULT_LOCK_TIMEOUT_SECONDS)
    try:
        timeout = float(value)
    except ValueError as error:
        raise ValueError("{} must be a non-negative number".format(LOCK_TIMEOUT_ENV)) from error
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError("{} must be a finite non-negative number".format(LOCK_TIMEOUT_ENV))
    return timeout


def _posix_try_lock(descriptor, fcntl_module=None):
    if fcntl_module is None:
        import fcntl as fcntl_module

    fcntl_module.flock(descriptor, fcntl_module.LOCK_EX | fcntl_module.LOCK_NB)


def _posix_unlock(descriptor, fcntl_module=None):
    if fcntl_module is None:
        import fcntl as fcntl_module

    fcntl_module.flock(descriptor, fcntl_module.LOCK_UN)


def _windows_try_lock(descriptor, msvcrt_module=None, seek=None):
    if msvcrt_module is None:
        import msvcrt as msvcrt_module

    seek = os.lseek if seek is None else seek
    seek(descriptor, 0, os.SEEK_SET)
    msvcrt_module.locking(descriptor, msvcrt_module.LK_NBLCK, 1)


def _windows_unlock(descriptor, msvcrt_module=None, seek=None):
    if msvcrt_module is None:
        import msvcrt as msvcrt_module

    seek = os.lseek if seek is None else seek
    seek(descriptor, 0, os.SEEK_SET)
    msvcrt_module.locking(descriptor, msvcrt_module.LK_UNLCK, 1)


def _try_advisory_lock(descriptor):
    if os.name == "nt":
        _windows_try_lock(descriptor)
    else:
        _posix_try_lock(descriptor)


def _release_advisory_lock(descriptor):
    if os.name == "nt":
        _windows_unlock(descriptor)
    else:
        _posix_unlock(descriptor)


def _lock_is_busy(error):
    return isinstance(error, BlockingIOError) or getattr(error, "errno", None) in (
        errno.EACCES,
        errno.EAGAIN,
        errno.EDEADLK,
    )


def _is_reparse_point(stat_result):
    file_attributes = getattr(stat_result, "st_file_attributes", 0)
    reparse_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(file_attributes & reparse_attribute)


def _same_file(first, second):
    return (first.st_dev, first.st_ino) == (second.st_dev, second.st_ino)


def _write_all(descriptor, data):
    offset = 0
    while offset < len(data):
        written = os.write(descriptor, data[offset:])
        if written <= 0:
            raise OSError("short write while creating cache lock")
        offset += written


def _read_open_regular(descriptor, display_path, expected_size=None, max_size=None, expected_stat=None):
    opened = os.fstat(descriptor)
    if not stat.S_ISREG(opened.st_mode) or _is_reparse_point(opened):
        raise CachePathError("cache file is not regular: {}".format(display_path))
    if expected_stat is not None and not _same_file(expected_stat, opened):
        raise CachePathError("cache file changed while opening: {}".format(display_path))
    size = opened.st_size
    if not isinstance(size, int) or isinstance(size, bool) or size < 0 or size > sys.maxsize:
        raise CacheIntegrityError("cache file has an invalid size: {}".format(display_path))
    if expected_size is not None and size != expected_size:
        raise CacheIntegrityError("cache file size does not match its manifest: {}".format(display_path))
    if max_size is not None and size > max_size:
        raise CacheIntegrityError("cache file exceeds its size limit: {}".format(display_path))

    contents = bytearray(size)
    view = memoryview(contents)
    offset = 0
    with os.fdopen(descriptor, "rb", closefd=False) as cache_file:
        while offset < size:
            read = cache_file.readinto(view[offset:])
            if not read:
                raise CacheIntegrityError("cache file ended before its declared size: {}".format(display_path))
            offset += read
        if cache_file.read(1):
            raise CacheIntegrityError("cache file grew while it was read: {}".format(display_path))
    after = os.fstat(descriptor)
    if not _same_file(opened, after) or after.st_size != size:
        raise CacheIntegrityError("cache file changed while it was read: {}".format(display_path))
    return contents, opened


def _read_regular_file(path, expected_size=None, max_size=None):
    """Reads one regular file without following a symlink into native code."""

    path = Path(path)
    before = os.lstat(os.fspath(path))
    if stat.S_ISLNK(before.st_mode) or _is_reparse_point(before) or not stat.S_ISREG(before.st_mode):
        raise CachePathError("cache file is not a regular non-link file: {}".format(path))

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(os.fspath(path), flags)
    try:
        return _read_open_regular(
            descriptor,
            path,
            expected_size=expected_size,
            max_size=max_size,
            expected_stat=before,
        )
    finally:
        os.close(descriptor)


def _read_regular_at(directory_fd, basename, expected_size=None, max_size=None):
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(basename, flags, dir_fd=directory_fd)
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise CachePathError("cache file is a link: {}".format(basename)) from error
        raise
    try:
        return _read_open_regular(
            descriptor,
            basename,
            expected_size=expected_size,
            max_size=max_size,
        )
    finally:
        os.close(descriptor)


def _remove_flat_bundle(path, best_effort=False):
    if os.name == "nt":
        if best_effort:
            return False
        raise CacheError("automatic cache residual cleanup is disabled on Windows")
    try:
        return _remove_flat_bundle_impl(path, best_effort)
    except (OSError, CachePathError):
        if best_effort:
            return False
        raise


def _remove_flat_bundle_impl(path, best_effort):
    """Removes only a flat cache directory's known entries without recursion."""

    path = Path(path)
    try:
        path_stat = os.lstat(os.fspath(path))
    except FileNotFoundError:
        return True
    if stat.S_ISLNK(path_stat.st_mode) or _is_reparse_point(path_stat):
        if best_effort:
            return False
        raise CachePathError("refusing to remove cache link: {}".format(path))
    if not stat.S_ISDIR(path_stat.st_mode):
        if best_effort:
            return False
        raise CachePathError("refusing to remove non-directory cache bundle: {}".format(path))

    entries = list(os.scandir(os.fspath(path)))
    for entry in entries:
        entry_stat = entry.stat(follow_symlinks=False)
        is_link = stat.S_ISLNK(entry_stat.st_mode)
        if _is_reparse_point(entry_stat) or (stat.S_ISDIR(entry_stat.st_mode) and not is_link):
            return False
        if not is_link and (entry.name not in _BUNDLE_FILENAMES or not stat.S_ISREG(entry_stat.st_mode)):
            return False

    directory_fd = None
    if os.name != "nt":
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            directory_fd = os.open(os.fspath(path), flags)
            if not _same_file(path_stat, os.fstat(directory_fd)):
                raise CachePathError("cache bundle changed before cleanup: {}".format(path))
        except (OSError, CachePathError):
            if directory_fd is not None:
                os.close(directory_fd)
            if best_effort:
                return False
            raise
    try:
        for entry in entries:
            try:
                if directory_fd is not None:
                    os.unlink(entry.name, dir_fd=directory_fd)
                else:
                    Path(entry.path).unlink()
            except FileNotFoundError:
                pass
        os.rmdir(os.fspath(path))
        return True
    except (OSError, CachePathError):
        if best_effort:
            return False
        raise
    finally:
        if directory_fd is not None:
            os.close(directory_fd)


class CacheBundle:
    """One immutable tagger/verbalizer cache bundle.

    ``cache_root`` is an explicitly trusted user-controlled root. It is
    resolved once; every generated child stays beneath it and must be a normal
    directory or regular file, never a symlink, junction, or special file.
    """

    def __init__(self, cache_root, prefix, ordertype, cache_config, source_fingerprint):
        if (not isinstance(prefix, str) or prefix in (".", "..") or not _SAFE_PREFIX.fullmatch(prefix)
                or Path(prefix).name != prefix):
            raise ValueError("cache prefix must be one safe path component")

        self.cache_root = Path(cache_root).expanduser().resolve(strict=False)
        self.prefix = prefix
        self.ordertype = ordertype
        self.cache_config = cache_config
        self.source_fingerprint = source_fingerprint
        self.builder = _builder_identity()

        config_identity = {
            "config": cache_config,
            "ordertype": ordertype,
            "prefix": prefix,
        }
        self.config_digest = _hash_bytes(_canonical_json(config_identity))
        bundle_identity = {
            "builder": self.builder,
            "config_digest": self.config_digest,
            "source_fingerprint": source_fingerprint,
        }
        self.bundle_digest = _hash_bytes(_canonical_json(bundle_identity))
        self.parent = self.cache_root / prefix / self.config_digest
        self.path = self.parent / self.bundle_digest
        self.lock_path = self.parent / ".{}.lock".format(self.bundle_digest)
        for path in (self.parent, self.path, self.lock_path):
            self._assert_under_root(path)

    @property
    def tagger_path(self):
        return self.path / _BUNDLE_FILENAMES[0]

    @property
    def verbalizer_path(self):
        return self.path / _BUNDLE_FILENAMES[1]

    @property
    def manifest_path(self):
        return self.path / _BUNDLE_FILENAMES[2]

    def _assert_under_root(self, path):
        root = os.path.normcase(os.path.abspath(os.fspath(self.cache_root)))
        candidate = os.path.normcase(os.path.abspath(os.fspath(path)))
        try:
            contained = os.path.commonpath((root, candidate)) == root
        except ValueError:
            contained = False
        if not contained:
            raise CachePathError("cache path escapes its trusted root: {}".format(path))

    def _ensure_directory(self, path, mode=0o700):
        self._assert_under_root(path)
        try:
            os.mkdir(os.fspath(path), mode)
        except FileExistsError:
            pass
        path_stat = os.lstat(os.fspath(path))
        if stat.S_ISLNK(path_stat.st_mode) or _is_reparse_point(path_stat) or not stat.S_ISDIR(path_stat.st_mode):
            raise CachePathError("cache directory is not a normal directory: {}".format(path))

    def _validate_directory_chain(self, path):
        """Validates every cache-owned ancestor without following a link."""

        self._assert_under_root(path)
        relative = Path(path).relative_to(self.cache_root)
        current = self.cache_root
        candidates = [current]
        for component in relative.parts:
            current = current / component
            candidates.append(current)

        for candidate in candidates:
            candidate_stat = os.lstat(os.fspath(candidate))
            if (stat.S_ISLNK(candidate_stat.st_mode) or _is_reparse_point(candidate_stat)
                    or not stat.S_ISDIR(candidate_stat.st_mode)):
                raise CachePathError("cache ancestor is not a normal directory: {}".format(candidate))
            self._assert_under_root(Path(candidate).resolve(strict=True))

    def _open_directory_chain(self, path):
        """Binds each POSIX cache ancestor with O_NOFOLLOW directory fds."""

        self._assert_under_root(path)
        relative = Path(path).relative_to(self.cache_root)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(os.fspath(self.cache_root), flags)
        try:
            root_stat = os.fstat(descriptor)
            if not stat.S_ISDIR(root_stat.st_mode) or _is_reparse_point(root_stat):
                raise CachePathError("cache root is not a normal directory: {}".format(self.cache_root))
            for component in relative.parts:
                try:
                    child = os.open(component, flags, dir_fd=descriptor)
                except OSError as error:
                    if error.errno in (errno.ELOOP, errno.ENOTDIR):
                        raise CachePathError("cache ancestor is not a normal directory: {}".format(component)) from error
                    raise
                try:
                    child_stat = os.fstat(child)
                    if not stat.S_ISDIR(child_stat.st_mode) or _is_reparse_point(child_stat):
                        raise CachePathError("cache ancestor is not a normal directory: {}".format(component))
                except BaseException:
                    os.close(child)
                    raise
                os.close(descriptor)
                descriptor = child
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def _ensure_parent(self):
        # The resolved root is trusted, but children created by the cache are
        # checked on every use to reject link/junction substitution.
        self.cache_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        root_stat = os.lstat(os.fspath(self.cache_root))
        if stat.S_ISLNK(root_stat.st_mode) or _is_reparse_point(root_stat) or not stat.S_ISDIR(root_stat.st_mode):
            raise CachePathError("cache root is not a normal directory: {}".format(self.cache_root))
        self._ensure_directory(self.cache_root / self.prefix)
        self._ensure_directory(self.parent)

    def _expected_manifest_identity(self):
        return {
            "builder": self.builder,
            "bundle_digest": self.bundle_digest,
            "cache_format": CACHE_FORMAT_VERSION,
            "config": self.cache_config,
            "config_digest": self.config_digest,
            "ordertype": self.ordertype,
            "prefix": self.prefix,
            "source_fingerprint": self.source_fingerprint,
        }

    def _check_bundle_directory(self, bundle_path):
        self._validate_directory_chain(bundle_path)

    def _load_path(self, bundle_path):
        bundle_fd = None
        try:
            if os.name == "nt":
                self._check_bundle_directory(bundle_path)
                entries = {entry.name for entry in os.scandir(os.fspath(bundle_path))}

                def read_file(basename, **kwargs):
                    return _read_regular_file(bundle_path / basename, **kwargs)

            else:
                bundle_fd = self._open_directory_chain(bundle_path)
                entries = set(os.listdir(bundle_fd))

                def read_file(basename, **kwargs):
                    return _read_regular_at(bundle_fd, basename, **kwargs)

            if entries != set(_BUNDLE_FILENAMES):
                return None

            try:
                manifest_bytes, _ = read_file("manifest.json", max_size=_MANIFEST_MAX_BYTES)
                manifest = json.loads(manifest_bytes.decode("utf-8"))
            except FileNotFoundError:
                return None
            except (OSError, UnicodeError, ValueError):
                return None
            if not isinstance(manifest, dict):
                return None

            expected = self._expected_manifest_identity()
            if any(manifest.get(key) != value for key, value in expected.items()):
                return None
            files = manifest.get("files")
            if not isinstance(files, dict) or set(files) != {"tagger.fst", "verbalizer.fst"}:
                return None

            graphs = []
            for basename in ("tagger.fst", "verbalizer.fst"):
                metadata = files.get(basename)
                if not isinstance(metadata, dict):
                    return None
                expected_size = metadata.get("size")
                if (not isinstance(expected_size, int) or isinstance(expected_size, bool) or expected_size < 0
                        or expected_size > sys.maxsize):
                    return None
                try:
                    fst_bytes, _ = read_file(
                        basename,
                        expected_size=expected_size,
                        max_size=_FST_MAX_BYTES,
                    )
                except FileNotFoundError:
                    return None
                except CacheIntegrityError:
                    return None
                except CachePathError:
                    raise
                except OSError:
                    return None
                try:
                    if metadata.get("sha256") != hashlib.sha256(fst_bytes).hexdigest():
                        return None
                    graphs.append(Fst.read_from_string(fst_bytes).optimize())
                except (RuntimeError, TypeError, ValueError):
                    return None
            return tuple(graphs)
        except FileNotFoundError:
            return None
        finally:
            if bundle_fd is not None:
                os.close(bundle_fd)

    def load(self):
        """Loads only a complete, matching, checksummed bundle."""

        return self._load_path(self.path)

    def _open_lock_anchor(self):
        self._ensure_parent()
        if os.name == "nt" and os.path.lexists(os.fspath(self.lock_path)):
            anchor_stat = os.lstat(os.fspath(self.lock_path))
            if (stat.S_ISLNK(anchor_stat.st_mode) or _is_reparse_point(anchor_stat) or not stat.S_ISREG(anchor_stat.st_mode)):
                raise CachePathError("cache lock anchor is not a regular file: {}".format(self.lock_path))

        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_BINARY", 0)
        if os.name != "nt":
            flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(os.fspath(self.lock_path), flags, 0o600)
        try:
            anchor_stat = os.fstat(descriptor)
            if not stat.S_ISREG(anchor_stat.st_mode) or _is_reparse_point(anchor_stat):
                raise CachePathError("cache lock anchor is not a regular file: {}".format(self.lock_path))
            if anchor_stat.st_size == 0:
                os.lseek(descriptor, 0, os.SEEK_SET)
                _write_all(descriptor, b"\0")
                os.fsync(descriptor)
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    @staticmethod
    def _read_owner_diagnostic(descriptor):
        try:
            size = os.fstat(descriptor).st_size
            payload_size = max(0, size - 1)
            if payload_size == 0:
                return "empty owner payload"
            if payload_size > _OWNER_PAYLOAD_MAX_BYTES:
                return "owner payload too large ({} bytes)".format(payload_size)
            os.lseek(descriptor, 1, os.SEEK_SET)
            payload_bytes = os.read(descriptor, payload_size)
            if len(payload_bytes) != payload_size:
                return "partial owner payload"
            payload = json.loads(payload_bytes.decode("utf-8"))
            if not isinstance(payload, dict):
                return "non-dict owner payload"
            return repr(payload)
        except (OSError, UnicodeError, ValueError, TypeError) as error:
            return "unreadable owner payload {!r}".format(error)

    @staticmethod
    def _write_owner_payload(descriptor, payload):
        encoded = _canonical_json(payload)
        if len(encoded) > _OWNER_PAYLOAD_MAX_BYTES:
            raise CacheError("cache lock owner payload exceeds {} bytes".format(_OWNER_PAYLOAD_MAX_BYTES))
        os.ftruncate(descriptor, 1)
        os.lseek(descriptor, 1, os.SEEK_SET)
        _write_all(descriptor, encoded)
        os.fsync(descriptor)

    @contextmanager
    def lock(self, timeout=None):
        timeout = _configured_lock_timeout() if timeout is None else float(timeout)
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("cache lock timeout must be a finite non-negative number")
        descriptor = self._open_lock_anchor()
        deadline = time.monotonic() + timeout
        acquired = False
        try:
            while not acquired:
                try:
                    _try_advisory_lock(descriptor)
                    acquired = True
                except OSError as error:
                    if not _lock_is_busy(error):
                        raise CacheError("failed to acquire OS cache lock: {}".format(self.lock_path)) from error
                    if time.monotonic() >= deadline:
                        owner = self._read_owner_diagnostic(descriptor)
                        raise CacheLockTimeout(self.lock_path, timeout, owner)
                    time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

            self._write_owner_payload(
                descriptor,
                {
                    "acquired": time.time(),
                    "hostname": platform.node(),
                    "token": uuid.uuid4().hex,
                },
            )
            try:
                yield
            finally:
                try:
                    _release_advisory_lock(descriptor)
                except OSError:
                    pass
                acquired = False
        finally:
            os.close(descriptor)

    def remove_invalid(self):
        """Quarantines exactly this non-link bundle under ``lock()``."""

        try:
            self._check_bundle_directory(self.path)
        except FileNotFoundError:
            return
        quarantine = self.parent / ".{}.corrupt-{}".format(self.bundle_digest, uuid.uuid4().hex)
        self._replace_with_retry(self.path, quarantine, "quarantining invalid cache bundle")
        _remove_flat_bundle(quarantine, best_effort=True)

    def recover_residuals(self):
        """Recovers a complete previous bundle, then cleans this exact key."""

        self._ensure_parent()
        if os.name == "nt":
            return
        prefixes = (
            ".{}.tmp-".format(self.bundle_digest),
            ".{}.previous-".format(self.bundle_digest),
            ".{}.corrupt-".format(self.bundle_digest),
        )
        residuals = []
        for child in self.parent.iterdir():
            if any(child.name.startswith(prefix) for prefix in prefixes):
                child_stat = os.lstat(os.fspath(child))
                if (not stat.S_ISLNK(child_stat.st_mode) and not _is_reparse_point(child_stat)
                        and stat.S_ISDIR(child_stat.st_mode)):
                    residuals.append(child)

        if not os.path.lexists(os.fspath(self.path)):
            previous = [path for path in residuals if path.name.startswith(prefixes[1])]
            previous.sort(key=lambda path: os.lstat(os.fspath(path)).st_mtime, reverse=True)
            for candidate in previous:
                if self._load_path(candidate) is not None:
                    self._replace_with_retry(candidate, self.path, "recovering previous cache bundle")
                    residuals.remove(candidate)
                    _fsync_directory(self.parent)
                    break

        for residual in residuals:
            _remove_flat_bundle(residual, best_effort=True)

    @staticmethod
    def _replace_with_retry(source, destination, operation):
        deadline = time.monotonic() + 2.0
        while True:
            try:
                os.replace(os.fspath(source), os.fspath(destination))
                return
            except PermissionError as error:
                if os.name != "nt" or time.monotonic() >= deadline:
                    raise CacheError("{} failed because Windows still has a cache reader open: {}".format(operation,
                                                                                                          source)) from error
                time.sleep(0.05)

    def publish(self, tagger, verbalizer):
        """Publishes a complete bundle under ``lock()``.

        Replacing an existing directory has a short interval where the final
        name is absent, but it is never present with a mixed or partial pair.
        Readers that observe that interval miss, wait for the key lock, and
        retry the completed bundle.
        """

        self._ensure_parent()
        temporary = Path(tempfile.mkdtemp(
            prefix=".{}.tmp-".format(self.bundle_digest),
            dir=os.fspath(self.parent),
        ))
        previous = None
        try:
            tagger_path = temporary / "tagger.fst"
            verbalizer_path = temporary / "verbalizer.fst"
            self._write_fst(tagger, tagger_path)
            self._write_fst(verbalizer, verbalizer_path)

            files = {}
            for path in (tagger_path, verbalizer_path):
                fst_bytes, _ = _read_regular_file(path)
                Fst.read_from_string(fst_bytes)
                files[path.name] = {
                    "sha256": hashlib.sha256(fst_bytes).hexdigest(),
                    "size": len(fst_bytes),
                }

            manifest = self._expected_manifest_identity()
            manifest["files"] = files
            self._write_manifest(temporary / "manifest.json", manifest)
            _fsync_directory(temporary)

            if os.path.lexists(os.fspath(self.path)):
                self._check_bundle_directory(self.path)
                previous = self.parent / ".{}.previous-{}".format(self.bundle_digest, uuid.uuid4().hex)
                self._replace_with_retry(self.path, previous, "moving the current cache bundle")
            try:
                self._replace_with_retry(temporary, self.path, "publishing the new cache bundle")
                temporary = None
                _fsync_directory(self.parent)
            except BaseException:
                if previous is not None and os.path.lexists(os.fspath(previous)) and not os.path.lexists(os.fspath(self.path)):
                    self._replace_with_retry(previous, self.path, "restoring the previous cache bundle")
                    previous = None
                raise
            if previous is not None:
                _remove_flat_bundle(previous, best_effort=True)
        finally:
            if temporary is not None and os.path.lexists(os.fspath(temporary)):
                _remove_flat_bundle(temporary, best_effort=True)

    @staticmethod
    def _write_fst(graph, path):
        graph.optimize().write(os.fspath(path))
        with open(path, "rb+") as cache_file:
            os.fsync(cache_file.fileno())

    @staticmethod
    def _write_manifest(path, manifest):
        with open(path, "w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file, ensure_ascii=False, sort_keys=True)
            manifest_file.write("\n")
            manifest_file.flush()
            os.fsync(manifest_file.fileno())
