import hashlib
import struct
import numpy as np
from typing import Any, Iterable, Tuple, Mapping
from collections.abc import Mapping as ABMapping

Pair = Tuple[str, Any]

def _update_array(h, arr: np.ndarray) -> None:
    a = np.asarray(arr)
    canon_dt = a.dtype.newbyteorder('<')
    if a.dtype != canon_dt:
        a = a.astype(canon_dt, copy=False)
    a = np.ascontiguousarray(a)

    h.update(b'ND')
    h.update(a.dtype.str.encode('ascii'))
    h.update(struct.pack('!B', a.ndim))
    h.update(struct.pack('!' + 'Q'*a.ndim, *a.shape))
    h.update(memoryview(a))

def _update_scalar(h, v: Any) -> None:
    if isinstance(v, (np.bool_, bool)):
        h.update(b'B'); h.update(b'\x01' if bool(v) else b'\x00')
    elif isinstance(v, (np.integer, int)):
        h.update(b'I'); h.update(str(int(v)).encode('ascii')); h.update(b';')
    elif isinstance(v, (np.floating, float)):
        if isinstance(v, np.floating):
            h.update(b'F'); h.update(np.array(v).dtype.str.encode('ascii'))
        else:
            h.update(b'Fpy')
        h.update(struct.pack('!d', float(v)))
    elif isinstance(v, (bytes, bytearray, memoryview)):
        b = bytes(v)
        h.update(b'BY'); h.update(struct.pack('!Q', len(b))); h.update(b)
    elif isinstance(v, str):
        b = v.encode('utf-8')
        h.update(b'S'); h.update(struct.pack('!Q', len(b))); h.update(b)
    else:
        s = repr(v)
        h.update(b'R'); h.update(struct.pack('!Q', len(s))); h.update(s.encode('utf-8'))

def _update_obj(h, v: Any) -> None:
    if isinstance(v, np.ndarray):
        _update_array(h, v)
    elif isinstance(v, (list, tuple)):
        h.update(b'L'); h.update(struct.pack('!Q', len(v)))
        for x in v: _update_obj(h, x)
    elif isinstance(v, dict) or isinstance(v, ABMapping):
        items = sorted(v.items(), key=lambda kv: kv[0])  # canonicalize dicts
        h.update(b'D'); h.update(struct.pack('!Q', len(items)))
        for k, x in items:
            _update_obj(h, k)
            _update_obj(h, x)
    else:
        _update_scalar(h, v)

def hash_conditions(conditions: Mapping[str, Any] | Iterable[Pair], digest_bytes: int = 16) -> str:
    """
    Deterministic hash for your conditions. Accepts either a dict-like mapping
    (hashed in sorted-key order) or an iterable of (key, value) pairs.
    """
    h = hashlib.blake2b(digest_size=digest_bytes)
    h.update(b'v1')

    if isinstance(conditions, ABMapping) or isinstance(conditions, dict):
        # Canonical for dicts: sort by key
        for k, v in sorted(conditions.items(), key=lambda kv: kv[0]):
            _update_obj(h, k)
            _update_obj(h, v)
    else:
        # Iterable of pairs: if you want order-agnostic behavior here too,
        # sort it; otherwise we preserve the given order for sequences.
        for k, v in conditions:
            _update_obj(h, k)
            _update_obj(h, v)

    return h.hexdigest()
