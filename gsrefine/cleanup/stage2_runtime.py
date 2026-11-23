"""Scope geometry reuse and nested wall-clock counters to one Stage-2 invocation."""
from contextlib import contextmanager
import hashlib
from pathlib import Path
from contextvars import ContextVar
from functools import wraps
from time import perf_counter

_active = ContextVar('stage2_run', default=None)


def run_scope(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        token = _active.set(dict(cache={}, timers={}, stack=[], start=perf_counter()))
        try:
            return fn(*args, **kwargs)
        finally:
            _active.reset(token)
    return wrapped


@contextmanager
def timed(name):
    state = _active.get()
    if state is None:
        yield
        return
    stack = state['stack']
    stack.append(0.)
    start = perf_counter()
    try:
        yield
    finally:
        elapsed = perf_counter() - start
        children = stack.pop()
        entry = state['timers'].setdefault(name, dict(calls=0, inclusive_s=0., exclusive_s=0.))
        entry['calls'] += 1
        entry['inclusive_s'] += elapsed
        entry['exclusive_s'] += elapsed - children
        if stack:
            stack[-1] += elapsed


def measured(name):
    def decorate(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            with timed(name):
                return fn(*args, **kwargs)
        return wrapped
    return decorate


def file_version(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    return (path, stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def file_digest(path):
    version = file_version(path)

    def read():
        with timed('read_file_sha256'):
            digest = hashlib.sha256()
            with version[0].open('rb') as stream:
                for chunk in iter(lambda: stream.read(4 << 20), b''):
                    digest.update(chunk)
            if file_version(path) != version:
                raise ValueError(f'Source changed during SHA-256 calculation: {path}')
            return digest.hexdigest()

    return reuse(('sha256', version), read)


def snapshot():
    state = _active.get()
    total = perf_counter() - state['start']
    steps = state['timers']
    return dict(total_s=total, steps=steps, other_s=total-sum(v['exclusive_s'] for v in steps.values()),
                scope='onepass entry to report assembly; nested steps included once in exclusive_s')


def reuse(key, build):
    state = _active.get()
    if state is None:
        return build()
    cache = state['cache']
    if key not in cache:
        cache[key] = build()
    return cache[key]


def mesh_resource(name, mesh, build):
    # Retain the input object as well as the result for the lifetime of the run,
    # so Python cannot recycle its id for a different mesh.
    return reuse((name, id(mesh)), lambda: (mesh, build(mesh)))[1]
