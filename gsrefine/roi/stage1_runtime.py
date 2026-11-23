"""Per-invocation digest reuse and wall-clock counters for the confirmed ROI worker."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from time import perf_counter

from gsrefine.cleanup.stage2_runtime import file_digest, file_version

_run = ContextVar("stage1_run", default=None)


@contextmanager
def session(version=2):
    if version not in (1, 2):
        raise ValueError("Unknown Stage-1 optimization version")
    token = _run.set(dict(version=version, digests={}, inputs={}, hits=0, reads=0, bytes_read=0,
                          steps={}, stack=[]))
    try:
        yield
    finally:
        _run.reset(token)


def managed(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        with session(kwargs.get("optimization_version", 2) if kwargs.get("accelerate_stage1", True) else 1):
            return fn(*args, **kwargs)
    return wrapped


def active():
    return _run.get() is not None


def fast():
    state = _run.get()
    return state is not None and state["version"] == 2


@contextmanager
def timed(name):
    state = _run.get()
    if state is None:
        yield
        return
    stack = state["stack"]
    stack.append(0.0)
    start = perf_counter()
    try:
        yield
    finally:
        elapsed = perf_counter() - start
        children = stack.pop()
        if stack:
            stack[-1] += elapsed
        entry = state["steps"].setdefault(name, dict(calls=0, inclusive_s=0., exclusive_s=0.))
        entry["calls"] += 1
        entry["inclusive_s"] += elapsed
        entry["exclusive_s"] += elapsed - children


def digest(path):
    state = _run.get()
    if state is None:
        return file_digest(path)
    version = file_version(path)
    if fast() and version in state["digests"]:
        state["hits"] += 1
        return state["digests"][version]
    with timed("file_sha256_read"):
        result = file_digest(path)
        if file_version(path) != version:
            raise ValueError(f"File changed during Stage-1 hashing: {path}")
    state["reads"] += 1
    state["bytes_read"] += version[3]
    if fast():
        state["digests"][version] = result
    return result


def watch_inputs(paths):
    if fast():
        _run.get()["inputs"] = {p: file_version(p) for p in paths}


def check_inputs():
    if fast():
        for path, original in _run.get()["inputs"].items():
            if file_version(path) != original:
                raise ValueError(f"Stage-1 source changed during the run: {path}")


def report():
    state = _run.get()
    return dict(version=state["version"], steps=state["steps"], hash_cache_hits=state["hits"],
                digest_reads=state["reads"], digest_bytes=state["bytes_read"])
