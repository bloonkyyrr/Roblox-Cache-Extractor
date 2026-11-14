"""Read, repair and export Roblox model files (.rbxm / .rbxmx / .rbxl).

The binary codec follows the rbx-dom "Roblox Binary Model Format, Version 0"
specification:

* 32 byte file header (magic, signature, version, class count, instance count)
* chunks framed by name + compressed length + uncompressed length + reserved
* chunk payloads compressed with LZ4 or ZSTD (detected by the magic number)
* ``INST`` carries the class id, class name, object format, instance count,
  referent array and inline property values
* ``PROP`` carries an array of one property for a whole class
* ``PRNT`` is ``u8`` version + count + child referent array + parent referent
  array
* referent arrays are byte interleaved, zigzag transformed and accumulated
* ``Float32`` is big endian with the sign bit moved to the least significant bit
* ``Int32``/``Enum`` are big endian zigzag transformed
* ``END`` carries the magic value ``</roblox>``

Roblox's cache delivers models with extra ``HASH``/``SIGU``/``SIGN`` chunks.
Studio refuses files that carry them, so :func:`sanitize_rbxm` rewrites a model
with only the standard chunks, which is what the extractor writes to disk.
"""

from __future__ import annotations

import math
import os
import re
import struct
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "RBXMError",
    "Instance",
    "MeshPrimitive",
    "read_model",
    "collect_meshes",
    "sanitize_rbxm",
    "sanitize_model_file",
    "write_model",
    "write_placeholder_model",
    "export_obj",
    "export_fbx",
    "export_model",
]

MAGIC = b"<roblox!"
SIGNATURE = b"\x89\xff\r\n\x1a\n"
HEADER_LENGTH = len(MAGIC) + len(SIGNATURE) + 2 + 4 + 4 + 8
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
END_MAGIC = b"</roblox>"

STANDARD_CHUNKS = {b"META", b"SSTR", b"INST", b"PROP", b"PRNT"}


class RBXMError(Exception):
    """Raised when a model file cannot be read or written."""


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------

def _lz4_block_decompress(src: bytes, expected_size: int) -> bytes:
    """Pure python LZ4 block decoder (used when the lz4 package is missing)."""
    dst = bytearray()
    i = 0
    n = len(src)
    while i < n:
        token = src[i]
        i += 1
        literal_len = token >> 4
        if literal_len == 15:
            while i < n:
                extra = src[i]
                i += 1
                literal_len += extra
                if extra != 255:
                    break
        if literal_len:
            dst += src[i:i + literal_len]
            i += literal_len
        if i >= n:
            break
        if i + 1 >= n:
            raise RBXMError("corrupt LZ4 stream (offset)")
        offset = src[i] | (src[i + 1] << 8)
        i += 2
        if offset == 0 or offset > len(dst):
            raise RBXMError("corrupt LZ4 stream (bad match offset)")
        match_len = token & 0x0F
        if match_len == 15:
            while i < n:
                extra = src[i]
                i += 1
                match_len += extra
                if extra != 255:
                    break
        match_len += 4
        start = len(dst) - offset
        for k in range(match_len):
            dst.append(dst[start + k])
    if expected_size and len(dst) < expected_size:
        raise RBXMError("truncated LZ4 chunk")
    return bytes(dst[:expected_size]) if expected_size else bytes(dst)


def _decompress_chunk(payload: bytes, uncompressed_size: int) -> bytes:
    if payload[:4] == ZSTD_MAGIC:
        try:
            import zstandard  # type: ignore
        except ImportError as exc:
            raise RBXMError(
                "zstd compressed chunk: install the 'zstandard' package "
                "(pip install zstandard)"
            ) from exc
        return zstandard.ZstdDecompressor().decompress(
            payload, max_output_size=uncompressed_size or 0
        )
    try:
        import lz4.block  # type: ignore

        return lz4.block.decompress(payload, uncompressed_size=uncompressed_size)
    except ImportError:
        return _lz4_block_decompress(payload, uncompressed_size)


def _compress_chunk(payload: bytes) -> bytes:
    """Compress a chunk payload the same way Roblox does (ZSTD frame)."""
    if len(payload) < 1024:
        return b""
    try:
        import zstandard  # type: ignore
    except ImportError:
        return b""
    return zstandard.ZstdCompressor(level=3).compress(payload)


def _make_chunk(name: bytes, payload: bytes) -> bytes:
    compressed = _compress_chunk(payload)
    if compressed and len(compressed) < len(payload):
        return name + struct.pack("<III", len(compressed), len(payload), 0) + compressed
    return name + struct.pack("<III", 0, len(payload), 0) + payload


# ---------------------------------------------------------------------------
# Primitive encoding helpers
# ---------------------------------------------------------------------------

def _zigzag_decode(value: int) -> int:
    return (value >> 1) ^ -(value & 1)


def _zigzag_encode(value: int) -> int:
    return (value << 1) ^ (value >> 31) if value < 0 else (value << 1)


def _decode_i32(raw: bytes) -> int:
    return _zigzag_decode(int.from_bytes(raw, "big"))


def _encode_i32(value: int) -> bytes:
    return _zigzag_encode(value & 0xFFFFFFFF if value >= 0 else value).to_bytes(4, "big")


def _decode_f32(raw: bytes) -> float:
    bits = int.from_bytes(raw, "big")
    sign = bits & 1
    exponent = (bits >> 24) & 0xFF
    mantissa = (bits >> 1) & 0x7FFFFF
    ieee = (sign << 31) | (exponent << 23) | mantissa
    return struct.unpack("<f", struct.pack("<I", ieee))[0]


def _encode_f32(value: float) -> bytes:
    ieee = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    sign = (ieee >> 31) & 1
    exponent = (ieee >> 23) & 0xFF
    mantissa = ieee & 0x7FFFFF
    return (((exponent << 24) | (mantissa << 1) | sign) & 0xFFFFFFFF).to_bytes(4, "big")


def _deinterleave(raw: bytes, count: int, width: int) -> List[bytes]:
    return [bytes(raw[k * count + i] for k in range(width)) for i in range(count)]


def _interleave(blobs: Sequence[bytes]) -> bytes:
    if not blobs:
        return b""
    width = len(blobs[0])
    out = bytearray(width * len(blobs))
    for k in range(width):
        base = k * len(blobs)
        for i, blob in enumerate(blobs):
            out[base + i] = blob[k]
    return bytes(out)


def _decode_referents(raw: bytes, count: int) -> List[int]:
    values = [_decode_i32(blob) for blob in _deinterleave(raw, count, 4)]
    out: List[int] = []
    total = 0
    for value in values:
        total += value
        out.append(total)
    return out


def _encode_referents(values: Sequence[int]) -> bytes:
    deltas: List[int] = []
    previous = 0
    for value in values:
        deltas.append(value - previous)
        previous = value
    return _interleave([_encode_i32(delta) for delta in deltas])


def _pack_string(text: str) -> bytes:
    encoded = text.encode("utf-8")
    return struct.pack("<I", len(encoded)) + encoded


class _Reader:
    __slots__ = ("data", "pos")

    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def remaining(self) -> int:
        return len(self.data) - self.pos

    def take(self, count: int) -> bytes:
        end = self.pos + count
        if count < 0 or end > len(self.data):
            raise RBXMError("unexpected end of data")
        chunk = self.data[self.pos:end]
        self.pos = end
        return chunk

    def u8(self) -> int:
        return self.take(1)[0]

    def u16(self) -> int:
        return struct.unpack("<H", self.take(2))[0]

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def string(self) -> str:
        length = self.u32()
        return self.take(length).decode("utf-8", "replace") if length else ""


# ---------------------------------------------------------------------------
# CFrame rotation table (IDs used when only the position is stored)
# ---------------------------------------------------------------------------

_CFRAME_ROTATIONS: Dict[int, Tuple[float, float, float]] = {
    0x02: (0, 0, 0),
    0x03: (90, 0, 0),
    0x05: (0, 180, 180),
    0x06: (-90, 0, 0),
    0x07: (0, 180, 90),
    0x09: (0, 90, 90),
    0x0A: (0, 0, 90),
    0x0C: (0, -90, 90),
    0x0D: (-90, -90, 0),
    0x0E: (0, -90, 0),
    0x10: (90, -90, 0),
    0x11: (0, 90, 180),
    0x14: (0, 180, 0),
    0x15: (-90, -180, 0),
    0x17: (0, 0, 180),
    0x18: (90, 180, 0),
    0x19: (0, 0, -90),
    0x1B: (0, -90, -90),
    0x1C: (0, -180, -90),
    0x1E: (0, 90, -90),
    0x1F: (90, 90, 0),
    0x20: (0, 90, 0),
    0x22: (-90, 90, 0),
    0x23: (0, -90, 180),
}


def _rotation_from_angles(angles_degrees: Tuple[float, float, float]) -> Tuple[float, ...]:
    """Build a CFrame rotation matrix from angles applied in Y -> X -> Z order."""
    x, y, z = (math.radians(value) for value in angles_degrees)
    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    cz, sz = math.cos(z), math.sin(z)

    # R = Rz * Rx * Ry
    return (
        cy * cz + sy * sx * sz, -cx * sz, -sy * cz + cy * sx * sz,
        cy * sz - sy * sx * cz, cx * cz, -sy * sz - cy * sx * cz,
        sy * cx, sx, cy * cx,
    )


def _identity_rotation() -> Tuple[float, ...]:
    return (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def _decode_cframe(raw: bytes) -> Optional[Tuple[float, ...]]:
    """Decode one CFrame value (12 floats: 9 rotation + 3 position)."""
    if not raw:
        return None
    rotation_id = raw[0]
    if rotation_id == 0:
        if len(raw) < 1 + 36 + 12:
            return None
        rotation = struct.unpack("<9f", raw[1:37])
        position_bytes = raw[37:49]
    else:
        if len(raw) < 13:
            return None
        rotation = _rotation_from_angles(_CFRAME_ROTATIONS[rotation_id]) if rotation_id in _CFRAME_ROTATIONS else _identity_rotation()
        position_bytes = raw[1:13]
    position = tuple(_decode_f32(position_bytes[i * 4:i * 4 + 4]) for i in range(3))
    return tuple(rotation) + position


def _read_cframe_values(reader: _Reader, count: int) -> List[bytes]:
    """Read ``count`` CFrames: sequential id/rotation, then a Vector3 array."""
    heads: List[bytes] = []
    for _ in range(count):
        rotation_id = reader.u8()
        if rotation_id == 0:
            heads.append(bytes([rotation_id]) + reader.take(36))
        else:
            heads.append(bytes([rotation_id]))

    # Positions are a Vector3 array: three separately interleaved f32 arrays.
    columns = [_deinterleave(reader.take(4 * count), count, 4) for _ in range(3)]
    return [head + b"".join(column[index] for column in columns) for index, head in enumerate(heads)]


# ---------------------------------------------------------------------------
# Property layouts
# ---------------------------------------------------------------------------
# Each entry lists the component arrays of the value, in file order.  Every
# component array is byte interleaved across instances in a PROP chunk.
# kind: (name, width).  ``f32``/``i32`` are transformed, ``raw`` kept as is.

_T = {
    "bool": ("raw", 1),
    "int": ("i32", 4),
    "float": ("f32", 4),
    "double": ("raw", 8),
    "u8": ("raw", 1),
    "i16": ("raw", 2),
    "u32": ("raw", 4),
    "u64": ("raw", 8),
    "ref": ("i32", 4),
}

_TYPE_LAYOUTS: Dict[int, List[Tuple[str, int]]] = {
    0x02: [_T["bool"]],
    0x03: [_T["int"]],
    0x04: [_T["float"]],
    0x05: [_T["double"]],
    0x06: [_T["float"], _T["int"]],                          # UDim
    0x07: [_T["float"], _T["float"], _T["int"], _T["int"]],    # UDim2 Xs Ys Xo Yo
    0x08: [_T["float"]] * 6,                                  # Ray
    0x09: [_T["u8"]],                                         # Faces
    0x0A: [_T["u8"]],                                         # Axes
    0x0B: [_T["u32"]],                                        # BrickColor
    0x0C: [_T["float"]] * 3,                                  # Color3
    0x0D: [_T["float"]] * 2,                                  # Vector2
    0x0E: [_T["float"]] * 3,                                  # Vector3
    0x0F: [_T["i16"]] * 2,                                    # Vector2int16
    0x11: [_T["float"]] * 4,                                  # Quaternion (legacy)
    0x12: [_T["int"]],                                        # Enum
    0x13: [_T["ref"]],                                        # Referent
    0x14: [_T["i16"]] * 3,                                    # Vector3int16
    0x17: [_T["float"]] * 2,                                  # NumberRange
    0x18: [_T["float"]] * 4,                                  # Rect
    0x1A: [_T["u8"]] * 3,                                     # Color3uint8
    0x1B: [("raw", 8)],                                       # Int64
    0x1C: [_T["u32"]],                                        # SharedString index
    0x21: [_T["u64"]],                                        # SecurityCapabilities
}

_T_STRING = 0x01
_T_INT32 = 0x03
_T_FLOAT32 = 0x04
_T_VECTOR3 = 0x0E
_T_CFRAME = 0x10
_T_ENUM = 0x12
_T_REFERENT = 0x13
_T_COLOR3 = 0x0C
_T_COLOR3UINT8 = 0x1A
_T_SHAREDSTRING = 0x1C
_T_OPTIONAL_CFRAME = 0x1E


def _read_var_value(reader: _Reader, type_id: int) -> bytes:
    """Read a variable length value (String, Content, sequences, ...)."""
    start = reader.pos
    if type_id in (_T_STRING, 0x1D):  # String / Bytecode
        length = reader.u32()
        reader.take(length)
        return reader.data[start:reader.pos]
    if type_id == 0x20:  # Font: family string, weight (u16), style (u8), face string
        reader.string()
        reader.take(3)
        reader.string()
        return reader.data[start:reader.pos]
    if type_id == 0x1F:  # UniqueId
        reader.take(16)
        return reader.data[start:reader.pos]
    if type_id == 0x15:  # NumberSequence
        count = reader.u32()
        reader.take(12 * count)
        return reader.data[start:reader.pos]
    if type_id == 0x16:  # ColorSequence
        count = reader.u32()
        reader.take(20 * count)
        return reader.data[start:reader.pos]
    if type_id == 0x19:  # PhysicalProperties
        if reader.u8():
            reader.take(20)
        return reader.data[start:reader.pos]
    if type_id == 0x22:  # Attributes
        count = reader.u32()
        for _ in range(count):
            name_length = reader.u32()
            reader.take(name_length)
            inner = reader.u8()
            _read_value(reader, inner)
        return reader.data[start:reader.pos]
    raise RBXMError(f"unsupported property type 0x{type_id:02X}")


def _read_value(reader: _Reader, type_id: int) -> bytes:
    if type_id in (_T_CFRAME, _T_OPTIONAL_CFRAME):
        return _read_value_array(reader, type_id, 1)[0]
    layout = _TYPE_LAYOUTS.get(type_id)
    if layout is None:
        return _read_var_value(reader, type_id)
    return b"".join(reader.take(width) for _, width in layout)


def _read_value_array(reader: _Reader, type_id: int, count: int) -> List[bytes]:
    """Read ``count`` values of one property and return them split per value."""
    if type_id == _T_CFRAME:
        return _read_cframe_values(reader, count)
    if type_id == _T_OPTIONAL_CFRAME:
        reader.u8()  # inner type id (0x10)
        values = _read_cframe_values(reader, count)
        reader.u8()  # presence type id (0x02)
        presence = reader.take(count)
        identity = b"\x02" + b"\x00" * 12
        return [value if presence[index] else identity for index, value in enumerate(values)]
    layout = _TYPE_LAYOUTS.get(type_id)
    if layout is None:
        return [_read_var_value(reader, type_id) for _ in range(count)]

    columns: List[List[bytes]] = []
    for _kind, width in layout:
        raw = reader.take(width * count)
        columns.append(_deinterleave(raw, count, width))

    values: List[bytes] = []
    for index in range(count):
        values.append(b"".join(column[index] for column in columns))
    return values


def _encode_values(values: Sequence[bytes], type_id: int) -> bytes:
    """Encode per-instance value blobs into one interleaved property array."""
    if type_id == _T_CFRAME:
        heads: List[bytes] = []
        positions: List[bytes] = []
        for blob in values:
            if not blob:
                raise RBXMError("empty CFrame value")
            if blob[0] == 0:
                heads.append(blob[:37])
                positions.append(blob[37:49])
            else:
                heads.append(blob[:1])
                positions.append(blob[1:13])
        if any(len(position) != 12 for position in positions):
            raise RBXMError("truncated CFrame value")
        encoded = b"".join(heads)
        for component in range(3):
            encoded += _interleave([position[component * 4:component * 4 + 4] for position in positions])
        return encoded
    layout = _TYPE_LAYOUTS.get(type_id)
    if layout is None:
        return b"".join(values)

    out = bytearray()
    offset = 0
    for _kind, width in layout:
        column = [blob[offset:offset + width] for blob in values]
        offset += width
        out += _interleave(column)
    return bytes(out)


# ---------------------------------------------------------------------------
# Instances
# ---------------------------------------------------------------------------

@dataclass
class Instance:
    """A Roblox instance read from a model file."""

    class_name: str
    referent: int = -1
    props: Dict[str, Tuple[int, bytes]] = field(default_factory=dict)
    values: Dict[str, object] = field(default_factory=dict)
    children: List["Instance"] = field(default_factory=list)
    parent: Optional["Instance"] = None

    @property
    def name(self) -> str:
        value = self.values.get("Name")
        return value if isinstance(value, str) and value else self.class_name

    def get(self, prop: str, default=None):
        return self.values.get(prop, default)


def _decode_property(type_id: int, raw: bytes):
    try:
        if type_id == _T_STRING:
            length = struct.unpack_from("<I", raw, 0)[0]
            return raw[4:4 + length].decode("utf-8", "replace")
        if type_id == 0x02:
            return bool(raw[0])
        if type_id in (_T_INT32, _T_ENUM, _T_REFERENT):
            return _decode_i32(raw[:4])
        if type_id == _T_FLOAT32:
            return _decode_f32(raw[:4])
        if type_id == 0x05:
            return struct.unpack("<d", raw[:8])[0]
        if type_id == _T_VECTOR3:
            return tuple(_decode_f32(raw[i * 4:i * 4 + 4]) for i in range(3))
        if type_id == 0x0D:
            return tuple(_decode_f32(raw[i * 4:i * 4 + 4]) for i in range(2))
        if type_id in (_T_CFRAME, _T_OPTIONAL_CFRAME):
            return _decode_cframe(raw)
        if type_id == _T_COLOR3:
            return tuple(_decode_f32(raw[i * 4:i * 4 + 4]) for i in range(3))
        if type_id == _T_COLOR3UINT8:
            return (raw[0], raw[1], raw[2])
    except (struct.error, IndexError):
        return None
    return None


def _chunk_iter(data: bytes):
    """Yield ``(name, payload, compressed_length)`` for each chunk."""
    if not data.startswith(MAGIC) or len(data) < HEADER_LENGTH:
        raise RBXMError("not a binary Roblox model file")
    pos = HEADER_LENGTH
    while pos + 16 <= len(data):
        name = data[pos:pos + 4]
        compressed, uncompressed, _reserved = struct.unpack_from("<III", data, pos + 4)
        body_length = compressed or uncompressed
        body = data[pos + 16:pos + 16 + body_length]
        if len(body) != body_length:
            raise RBXMError("truncated chunk")
        pos += 16 + body_length
        payload = _decompress_chunk(body, uncompressed) if compressed else body
        yield name, payload, compressed
        if name == b"END\0":
            return
    raise RBXMError("model file has no END chunk")


def _parse_binary(data: bytes) -> Tuple[List[Instance], bool]:
    class_count, instance_count = struct.unpack_from("<II", data, 16)
    instances: Dict[int, Instance] = {}
    order: List[int] = []
    classes: Dict[int, List[int]] = {}
    parents: List[Tuple[int, int]] = []
    complete = True

    for name, payload, _compressed in _chunk_iter(data):
        reader = _Reader(payload)
        if name == b"INST":
            class_id = reader.u32()
            class_name = reader.string()
            object_format = reader.u8()
            count = reader.u32()
            referents = _decode_referents(reader.take(4 * count), count)
            if object_format == 1:
                reader.take(count)  # service markers
            classes[class_id] = referents
            for ref in referents:
                inst = Instance(class_name=class_name, referent=ref)
                instances[ref] = inst
                order.append(ref)
            try:
                for ref in referents:
                    inst = instances[ref]
                    while reader.remaining() > 0:
                        prop_name = reader.string()
                        if not prop_name:
                            break
                        type_id = reader.u8()
                        raw = _read_value(reader, type_id)
                        inst.props[prop_name] = (type_id, raw)
                        inst.values[prop_name] = _decode_property(type_id, raw)
            except RBXMError:
                complete = False
        elif name == b"PROP":
            class_id = reader.u32()
            prop_name = reader.string()
            type_id = reader.u8()
            referents = classes.get(class_id, [])
            try:
                values = _read_value_array(reader, type_id, len(referents))
            except RBXMError:
                complete = False
                continue
            for ref, raw in zip(referents, values):
                inst = instances.get(ref)
                if inst is None:
                    continue
                inst.props[prop_name] = (type_id, raw)
                inst.values[prop_name] = _decode_property(type_id, raw)
        elif name == b"PRNT":
            reader.u8()  # version
            count = reader.u32()
            children = _decode_referents(reader.take(4 * count), count)
            parent_refs = _decode_referents(reader.take(4 * count), count)
            parents.extend(zip(children, parent_refs))

    for child, parent in parents:
        inst = instances.get(child)
        if inst is None or parent < 0 or parent not in instances:
            continue
        inst.parent = instances[parent]
        instances[parent].children.append(inst)

    roots = [instances[ref] for ref in order if instances[ref].parent is None]
    if instance_count and len(instances) != instance_count:
        complete = False
    return roots, complete


def _parse_xml(data: bytes) -> List[Instance]:
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise RBXMError(f"invalid XML model: {exc}") from exc

    def parse_item(item) -> Instance:
        inst = Instance(class_name=item.get("class", "Instance"))
        props = item.find("Properties")
        if props is not None:
            for prop in props:
                name = prop.get("name", prop.tag)
                text = (prop.text or "").strip()
                if prop.tag == "string":
                    inst.values[name] = text
                elif prop.tag == "bool":
                    inst.values[name] = text.lower() == "true"
                elif prop.tag in ("int", "int64", "token"):
                    try:
                        inst.values[name] = int(text)
                    except ValueError:
                        pass
                elif prop.tag in ("float", "double"):
                    try:
                        inst.values[name] = float(text)
                    except ValueError:
                        pass
                elif prop.tag == "Vector3":
                    parts = [float(v) for v in text.split()]
                    if len(parts) == 3:
                        inst.values[name] = tuple(parts)
                elif prop.tag == "CoordinateFrame":
                    numbers = [float(v) for v in text.split()]
                    if len(numbers) == 12:
                        inst.values[name] = tuple(numbers)
        for child in item.findall("Item"):
            parsed = parse_item(child)
            parsed.parent = inst
            inst.children.append(parsed)
        return inst

    roots = [parse_item(item) for item in root.findall("Item")]
    if not roots and root.tag == "Item":
        roots.append(parse_item(root))
    return roots


def read_model(path: str) -> List[Instance]:
    """Read a model file and return its root instances."""
    with open(path, "rb") as handle:
        data = handle.read()
    return read_model_bytes(data)


def read_model_bytes(data: bytes) -> List[Instance]:
    if not data:
        raise RBXMError("file is empty")
    if data.startswith(MAGIC):
        roots, _complete = _parse_binary(data)
        return roots
    return _parse_xml(data)


# ---------------------------------------------------------------------------
# Model re-writing
# ---------------------------------------------------------------------------

def sanitize_rbxm(data: bytes) -> bytes:
    """Rewrite a binary model keeping only standard chunks.

    Roblox's cache ships models with ``HASH``/``SIGU``/``SIGN`` chunks which
    Studio refuses to load ("The file is corrupted").  Re-writing the same
    payloads without those chunks produces a plain model that Studio opens.
    """
    if not data.startswith(MAGIC) or len(data) < HEADER_LENGTH:
        raise RBXMError("not a binary Roblox model file")

    class_count, instance_count = struct.unpack_from("<II", data, 16)
    out = bytearray()
    out += MAGIC + SIGNATURE
    out += struct.pack("<H", 0)
    out += struct.pack("<II", class_count, instance_count)
    out += b"\x00" * 8

    written_end = False
    for name, payload, _compressed in _chunk_iter(data):
        if name == b"END\0":
            written_end = True
            break
        if name in STANDARD_CHUNKS:
            out += _make_chunk(name, payload)

    if not written_end:
        raise RBXMError("model file has no END chunk")
    out += _make_chunk(b"END\0", END_MAGIC)
    return bytes(out)


def sanitize_model_file(source: str, target: str) -> None:
    with open(source, "rb") as handle:
        data = handle.read()
    with open(target, "wb") as handle:
        handle.write(sanitize_rbxm(data))


def _remap_value(type_id: int, raw: bytes, mapping: Dict[int, int]) -> bytes:
    """Translate a referent property value to the referents of the new file."""
    if type_id == _T_REFERENT and len(raw) == 4:
        return _encode_i32(mapping.get(_decode_i32(raw), -1))
    return raw


def write_model(instances: Sequence[Instance], path: str) -> None:
    """Write instances as a minimal binary RBXM file (no signatures)."""
    flat: List[Instance] = []

    def walk(inst: Instance) -> None:
        for child in inst.children:
            walk(child)
        flat.append(inst)

    for root in instances:
        walk(root)
    if not flat:
        raise RBXMError("nothing to write")

    mapping = {inst.referent: index for index, inst in enumerate(flat)}
    class_names: List[str] = []
    groups: Dict[str, List[Instance]] = {}
    for inst in flat:
        if inst.class_name not in groups:
            groups[inst.class_name] = []
            class_names.append(inst.class_name)
        groups[inst.class_name].append(inst)

    out = bytearray()
    out += MAGIC + SIGNATURE
    out += struct.pack("<H", 0)
    out += struct.pack("<II", len(class_names), len(flat))
    out += b"\x00" * 8

    for class_index, class_name in enumerate(class_names):
        group = groups[class_name]
        chunk = bytearray()
        chunk += struct.pack("<I", class_index)
        chunk += _pack_string(class_name)
        chunk += b"\x00"
        chunk += struct.pack("<I", len(group))
        chunk += _encode_referents([mapping[inst.referent] for inst in group])

        first_props = group[0].props
        uniform = all(inst.props.keys() == first_props.keys() for inst in group)
        if not uniform:
            # Classes whose instances carry different property sets keep the
            # values inline, the way Roblox stores services.
            for inst in group:
                for prop_name, (type_id, raw) in inst.props.items():
                    chunk += _pack_string(prop_name)
                    chunk += bytes([type_id])
                    chunk += _remap_value(type_id, raw, mapping)
                chunk += b"\x00\x00\x00\x00"
        out += _make_chunk(b"INST", bytes(chunk))

        if uniform:
            # Roblox stores regular properties in one PROP chunk per property.
            for prop_name, (type_id, _raw) in first_props.items():
                values = [
                    _remap_value(type_id, inst.props[prop_name][1], mapping)
                    for inst in group
                ]
                body = bytearray()
                body += struct.pack("<I", class_index)
                body += _pack_string(prop_name)
                body += bytes([type_id])
                body += _encode_values(values, type_id)
                out += _make_chunk(b"PROP", bytes(body))

    prnt = bytearray()
    prnt += b"\x00"
    prnt += struct.pack("<I", len(flat))
    prnt += _encode_referents([mapping[inst.referent] for inst in flat])
    prnt += _encode_referents([
        mapping[inst.parent.referent] if inst.parent is not None else -1
        for inst in flat
    ])
    out += _make_chunk(b"PRNT", bytes(prnt))
    out += _make_chunk(b"END\0", END_MAGIC)

    with open(path, "wb") as handle:
        handle.write(bytes(out))


def write_placeholder_model(path: str, name: str) -> None:
    """Write a valid one-instance model, for payloads that are not models."""
    folder = Instance(class_name="Folder", referent=0)
    folder.props["Name"] = (_T_STRING, _pack_string(name))
    folder.values["Name"] = name
    write_model([folder], path)


# Classes that Roblox stores as asset payloads instead of insertable models.
WRAPPER_CLASSES = {
    "PartOperationAsset",
    "PartOperationAssetInternal",
    "HiddenSurfaceRemovalAsset",
    "MeshPartAsset",
    "Asset",
}

# Properties of a wrapper class that hold a nested model.
WRAPPER_MODEL_PROPS = ("ChildData", "MeshData")


def _renumber(roots: Sequence[Instance]) -> None:
    """Give every instance a fresh referent so separate models can be merged."""
    counter = 0

    def walk(inst: Instance) -> None:
        nonlocal counter
        inst.referent = counter
        counter += 1
        for child in inst.children:
            walk(child)

    for root in roots:
        walk(root)


def unwrap_instances(roots: Sequence[Instance], notes: Optional[List[str]] = None) -> Optional[List[Instance]]:
    """Replace asset payload wrappers with the insertable model they contain.

    Roblox stores CSG results in ``PartOperationAsset`` objects (and similar)
    whose ``ChildData`` property contains a complete model. Studio refuses to
    open those wrapper files, so the contained model is exported instead.

    Returns ``None`` when the payload holds no model at all, which means the
    caller should write a placeholder instance.
    """
    if not roots:
        return None
    if not any(root.class_name in WRAPPER_CLASSES for root in roots):
        return list(roots)

    merged: List[Instance] = []
    for root in roots:
        if root.class_name not in WRAPPER_CLASSES:
            merged.append(root)
            continue
        found = False
        for prop_name in WRAPPER_MODEL_PROPS:
            entry = root.props.get(prop_name)
            if not entry or entry[0] != _T_STRING:
                continue
            blob = entry[1][4:] if len(entry[1]) >= 4 else b""
            if not blob.startswith(MAGIC):
                continue
            try:
                inner, _complete = _parse_binary(blob)
            except RBXMError:
                continue
            if inner:
                merged.extend(inner)
                found = True
        if not found and notes is not None:
            notes.append(f"{root.class_name} contained no insertable model")

    if not merged:
        return None
    _renumber(merged)
    return merged


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

@dataclass
class MeshPrimitive:
    """A triangulated mesh in world space."""

    name: str
    vertices: List[Tuple[float, float, float]]
    faces: List[Tuple[int, int, int]]
    faces_normals: List[Tuple[float, float, float]] = field(default_factory=list)

    @property
    def triangle_count(self) -> int:
        return len(self.faces)


PART_CLASSES = {
    "Part",
    "MeshPart",
    "WedgePart",
    "CornerWedgePart",
    "TrussPart",
    "UnionOperation",
    "NegateOperation",
    "Seat",
    "VehicleSeat",
    "SpawnLocation",
    "FlagStand",
    "SkateboardPlatform",
}

BOX_CLASSES = {
    "MeshPart",
    "UnionOperation",
    "NegateOperation",
    "TrussPart",
    "CornerWedgePart",
}

PART_TYPE_BALL = 0
PART_TYPE_BLOCK = 1
PART_TYPE_CYLINDER = 2


def _box(size: Tuple[float, float, float]):
    x, y, z = (size[0] / 2.0, size[1] / 2.0, size[2] / 2.0)
    verts = [
        (-x, -y, -z), (x, -y, -z), (x, y, -z), (-x, y, -z),
        (-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z),
    ]
    faces = [
        (0, 3, 2), (0, 2, 1),
        (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4),
        (2, 3, 7), (2, 7, 6),
        (1, 2, 6), (1, 6, 5),
        (0, 4, 7), (0, 7, 3),
    ]
    return verts, faces


def _sphere(size: Tuple[float, float, float], segments: int = 24, rings: int = 12):
    rx, ry, rz = (size[0] / 2.0, size[1] / 2.0, size[2] / 2.0)
    verts: List[Tuple[float, float, float]] = []
    faces: List[Tuple[int, int, int]] = []
    for ring in range(rings + 1):
        phi = math.pi * ring / rings
        for seg in range(segments):
            theta = 2.0 * math.pi * seg / segments
            verts.append((
                rx * math.sin(phi) * math.cos(theta),
                ry * math.cos(phi),
                rz * math.sin(phi) * math.sin(theta),
            ))
    for ring in range(rings):
        for seg in range(segments):
            nxt = (seg + 1) % segments
            a = ring * segments + seg
            b = ring * segments + nxt
            c = (ring + 1) * segments + nxt
            d = (ring + 1) * segments + seg
            if ring != 0:
                faces.append((a, b, c))
            if ring != rings - 1:
                faces.append((a, c, d))
    return verts, faces


def _cylinder(size: Tuple[float, float, float], segments: int = 24):
    half_x = size[0] / 2.0
    ry, rz = size[1] / 2.0, size[2] / 2.0
    verts: List[Tuple[float, float, float]] = []
    for side in (-half_x, half_x):
        for seg in range(segments):
            theta = 2.0 * math.pi * seg / segments
            verts.append((side, ry * math.cos(theta), rz * math.sin(theta)))
    verts.append((-half_x, 0.0, 0.0))
    verts.append((half_x, 0.0, 0.0))
    center_left = len(verts) - 2
    center_right = len(verts) - 1
    faces: List[Tuple[int, int, int]] = []
    for seg in range(segments):
        nxt = (seg + 1) % segments
        faces.append((seg, nxt, segments + nxt))
        faces.append((seg, segments + nxt, segments + seg))
        faces.append((center_left, nxt, seg))
        faces.append((center_right, segments + seg, segments + nxt))
    return verts, faces


def _wedge(size: Tuple[float, float, float]):
    x, y, z = (size[0] / 2.0, size[1] / 2.0, size[2] / 2.0)
    verts = [
        (-x, -y, -z), (x, -y, -z),
        (-x, -y, z), (x, -y, z),
        (-x, y, -z), (x, y, -z),
    ]
    faces = [
        (0, 2, 3), (0, 3, 1),
        (0, 1, 5), (0, 5, 4),
        (4, 5, 3), (4, 3, 2),
        (0, 4, 2),
        (1, 3, 5),
    ]
    return verts, faces


def _transform(verts, rotation, position):
    r00, r01, r02, r10, r11, r12, r20, r21, r22 = rotation
    px, py, pz = position
    out = []
    for vx, vy, vz in verts:
        out.append((
            px + r00 * vx + r01 * vy + r02 * vz,
            py + r10 * vx + r11 * vy + r12 * vz,
            pz + r20 * vx + r21 * vy + r22 * vz,
        ))
    return out


def _face_normal(verts, face):
    ax, ay, az = verts[face[0]]
    bx, by, bz = verts[face[1]]
    cx, cy, cz = verts[face[2]]
    ux, uy, uz = bx - ax, by - ay, bz - az
    vx, vy, vz = cx - ax, cy - ay, cz - az
    nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if length <= 1e-12:
        return (0.0, 0.0, 1.0)
    return (nx / length, ny / length, nz / length)


def collect_meshes(roots: Iterable[Instance], notes: Optional[List[str]] = None) -> List[MeshPrimitive]:
    """Turn every part in the tree into a triangulated mesh."""
    meshes: List[MeshPrimitive] = []
    used_names: Dict[str, int] = {}
    approximated = 0

    def walk(inst: Instance) -> None:
        nonlocal approximated
        if inst.class_name in PART_CLASSES:
            cframe = inst.get("CFrame")
            size = inst.get("Size", inst.get("size"))
            if isinstance(cframe, tuple) and len(cframe) == 12 and isinstance(size, tuple) and len(size) == 3:
                if size[0] > 0 and size[1] > 0 and size[2] > 0:
                    shape = inst.get("shape", inst.get("Shape"))
                    if inst.class_name in BOX_CLASSES:
                        verts, faces = _box(size)
                        if inst.class_name == "MeshPart":
                            approximated += 1
                    elif inst.class_name == "WedgePart":
                        verts, faces = _wedge(size)
                    elif shape == PART_TYPE_BALL:
                        verts, faces = _sphere(size)
                    elif shape == PART_TYPE_CYLINDER:
                        verts, faces = _cylinder(size)
                    else:
                        verts, faces = _box(size)
                    placed = _transform(verts, cframe[:9], cframe[9:12])
                    base = inst.name
                    count = used_names.get(base, 0) + 1
                    used_names[base] = count
                    name = base if count == 1 else f"{base}_{count}"
                    meshes.append(MeshPrimitive(
                        name=name,
                        vertices=placed,
                        faces=faces,
                        faces_normals=[_face_normal(placed, face) for face in faces],
                    ))
        for child in inst.children:
            walk(child)

    for root in roots:
        walk(root)

    if notes is not None and approximated:
        notes.append(
            f"{approximated} MeshPart/union instance(s) exported as bounding boxes "
            "(the .rbxm format stores only the mesh asset id, not triangles)"
        )
    return meshes


# ---------------------------------------------------------------------------
# OBJ / FBX writers
# ---------------------------------------------------------------------------

def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.\- ]+", "_", name).strip()
    return cleaned or "mesh"


def export_obj(meshes: Iterable[MeshPrimitive], path: str) -> int:
    """Write meshes to a Wavefront OBJ file. Returns the triangle count."""
    triangles = 0
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Exported by Roblox Audio Extractor\n")
        offset = 1
        for mesh in meshes:
            handle.write(f"o {_safe_name(mesh.name)}\n")
            for vx, vy, vz in mesh.vertices:
                handle.write(f"v {vx:.6f} {vy:.6f} {vz:.6f}\n")
            for face in mesh.faces:
                a, b, c = (index + offset for index in face)
                handle.write(f"f {a} {b} {c}\n")
            offset += len(mesh.vertices)
            triangles += len(mesh.faces)
    return triangles


def export_fbx(meshes: Iterable[MeshPrimitive], path: str) -> int:
    """Write meshes to an ASCII FBX 7.3 file. Returns the triangle count."""
    meshes = list(meshes)
    triangles = sum(len(mesh.faces) for mesh in meshes)

    def fmt(values) -> str:
        return ",".join(f"{value:.6f}" for value in values)

    lines: List[str] = []
    add = lines.append
    add("; FBX 7.3.0 project file")
    add("; Exported by Roblox Audio Extractor")
    add("")
    add("FBXHeaderExtension:  {")
    add("\tFBXHeaderVersion: 1003")
    add("\tFBXVersion: 7300")
    add("}")
    add("GlobalSettings:  {")
    add("\tVersion: 1000")
    add("\tProperties70:  {")
    add('\t\tP: "UpAxis", "int", "Integer", "",1')
    add('\t\tP: "UpAxisSign", "int", "Integer", "",1')
    add('\t\tP: "FrontAxis", "int", "Integer", "",2')
    add('\t\tP: "FrontAxisSign", "int", "Integer", "",-1')
    add('\t\tP: "CoordAxis", "int", "Integer", "",0')
    add('\t\tP: "CoordAxisSign", "int", "Integer", "",1')
    add('\t\tP: "UnitScaleFactor", "double", "Number", "",1')
    add("\t}")
    add("}")
    add("Objects:  {")
    for index, mesh in enumerate(meshes):
        geometry_id = 100000 + index * 2
        model_id = geometry_id + 1
        name = _safe_name(mesh.name)
        add(f'\tGeometry: {geometry_id}, "Geometry::{name}", "Mesh" {{')
        flat = []
        for vx, vy, vz in mesh.vertices:
            flat.extend((vx, vy, vz))
        add(f"\t\tVertices: *{len(flat)} {{")
        add(f"\t\t\ta: {fmt(flat)}")
        add("\t\t}")
        polygon: List[str] = []
        normals: List[str] = []
        for face_index, face in enumerate(mesh.faces):
            a, b, c = face
            polygon.extend((str(a), str(b), str(-c - 1)))
            normal = mesh.faces_normals[face_index] if face_index < len(mesh.faces_normals) else (0.0, 0.0, 1.0)
            for _ in range(3):
                normals.extend(fmt(normal).split(","))
        add(f"\t\tPolygonVertexIndex: *{len(polygon)} {{")
        add(f"\t\t\ta: {','.join(polygon)}")
        add("\t\t}")
        add("\t\tGeometryVersion: 124")
        add("\t\tLayerElementNormal: 0 {")
        add("\t\t\tVersion: 101")
        add('\t\t\tName: ""')
        add('\t\t\tMappingInformationType: "ByPolygonVertex"')
        add('\t\t\tReferenceInformationType: "Direct"')
        add(f"\t\t\tNormals: *{len(normals)} {{")
        add(f"\t\t\t\ta: {','.join(normals)}")
        add("\t\t\t}")
        add("\t\t}")
        add("\t\tLayer: 0 {")
        add("\t\t\tVersion: 100")
        add("\t\t\tLayerElement:  {")
        add('\t\t\t\tType: "LayerElementNormal"')
        add("\t\t\t\tTypedIndex: 0")
        add("\t\t\t}")
        add("\t\t}")
        add("\t}")
        add(f'\tModel: {model_id}, "Model::{name}", "Mesh" {{')
        add("\t\tVersion: 232")
        add("\t\tProperties70:  {")
        add('\t\t\tP: "DefaultAttributeIndex", "int", "Integer", "",0')
        add("\t\t}")
        add("\t\tShading: T")
        add('\t\tCulling: "CullingOff"')
        add("\t}")
    add("}")
    add("Connections:  {")
    for index in range(len(meshes)):
        geometry_id = 100000 + index * 2
        model_id = geometry_id + 1
        add(f'\tC: "OO",{model_id},0')
        add(f'\tC: "OO",{geometry_id},{model_id}')
    add("}")

    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")
    return triangles


# ---------------------------------------------------------------------------
# High level export
# ---------------------------------------------------------------------------

def export_model(
    source_path: str,
    output_dir: str,
    export_rbxm_file: bool = True,
    export_obj_file: bool = False,
    export_fbx_file: bool = False,
) -> Dict[str, object]:
    """Export one model file as a Studio compatible RBXM and/or OBJ/FBX."""
    with open(source_path, "rb") as handle:
        data = handle.read()
    if not data:
        raise RBXMError("file is empty")

    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(source_path))[0]
    written: List[str] = []
    notes: List[str] = []

    is_binary = data.startswith(MAGIC)

    if export_rbxm_file:
        if is_binary:
            target = os.path.join(output_dir, f"{stem}.rbxm")
            try:
                roots = unwrap_instances(read_model_bytes(data), notes)
            except RBXMError as error:
                roots = None
                notes.append(f"{stem}: unreadable model, wrote a placeholder ({error})")
            if roots:
                try:
                    write_model(roots, target)
                except RBXMError as error:
                    write_placeholder_model(target, stem)
                    notes.append(f"{stem}: could not rebuild the model, wrote a placeholder ({error})")
            else:
                write_placeholder_model(target, stem)
            written.append(target)
        else:
            target = os.path.join(output_dir, f"{stem}.rbxmx")
            with open(target, "wb") as handle:
                handle.write(data)
            written.append(target)

    meshes: List[MeshPrimitive] = []
    if export_obj_file or export_fbx_file:
        roots = unwrap_instances(read_model_bytes(data), notes) or []
        meshes = collect_meshes(roots, notes)
        if not meshes:
            notes.append(f"{stem}: no parts with geometry were found")

    triangles = 0
    if export_obj_file:
        target = os.path.join(output_dir, f"{stem}.obj")
        triangles = export_obj(meshes, target)
        written.append(target)

    if export_fbx_file:
        target = os.path.join(output_dir, f"{stem}.fbx")
        if not triangles:
            triangles = sum(len(mesh.faces) for mesh in meshes)
        export_fbx(meshes, target)
        written.append(target)

    unique_notes: List[str] = []
    for note in notes:
        if note not in unique_notes:
            unique_notes.append(note)

    return {
        "source": source_path,
        "files": written,
        "parts": len(meshes),
        "triangles": triangles,
        "notes": unique_notes,
    }
