"""Read Roblox model files (.rbxm / .rbxmx), collect part geometry and export it.

Supported outputs:
  * .rbxm    - binary Roblox model containing the extracted geometry-bearing instances
  * .obj     - Wavefront OBJ (one object per part)
  * .fbx     - ASCII FBX 7.3 (one mesh model per part)

The reader understands the binary RBXM container (INST / PROP / PRNT / META /
SSTR chunks, LZ4 chunks and optional zstd chunks when the ``zstandard`` module
is available) and, as a fallback, the XML .rbxmx representation.

Property payloads are kept as their original bytes so that re-serialising a
model never loses data that we do not understand.
"""

from __future__ import annotations

import math
import os
import re
import struct
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

__all__ = [
    "RBXMError",
    "Instance",
    "MeshPrimitive",
    "read_model",
    "collect_meshes",
    "write_rbxm",
    "export_obj",
    "export_fbx",
    "export_model",
]

MAGIC = b"<roblox!"
SIGNATURE = b"\x89\xff\r\n\x1a\n"


class RBXMError(Exception):
    """Raised when a model file cannot be read or written."""


# ---------------------------------------------------------------------------
# LZ4 block decompression (Roblox compresses chunks with the LZ4 block format)
# ---------------------------------------------------------------------------

def _lz4_decompress(src: bytes, expected_size: int) -> bytes:
    dst = bytearray()
    i = 0
    n = len(src)
    while i < n:
        token = src[i]
        i += 1
        literal_len = token >> 4
        if literal_len == 15:
            while True:
                if i >= n:
                    raise RBXMError("corrupt LZ4 stream (literal length)")
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
            while True:
                if i >= n:
                    raise RBXMError("corrupt LZ4 stream (match length)")
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
    return bytes(dst[:expected_size] if expected_size else dst)


def _decompress_chunk(payload: bytes, uncompressed_size: int) -> bytes:
    if len(payload) >= 4 and payload[:4] == b"\x28\xb5\x2f\xfd":
        try:
            import zstandard  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RBXMError(
                "this model is zstd-compressed; install the 'zstandard' package to read it"
            ) from exc
        return zstandard.ZstdDecompressor().decompress(payload, max_output_size=uncompressed_size or 0)
    if len(payload) == uncompressed_size:
        return payload
    return _lz4_decompress(payload, uncompressed_size)


# ---------------------------------------------------------------------------
# Binary reader helpers
# ---------------------------------------------------------------------------

class _Reader:
    __slots__ = ("data", "pos")

    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def remaining(self) -> int:
        return len(self.data) - self.pos

    def take(self, count: int) -> bytes:
        end = self.pos + count
        if end > len(self.data):
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

    def i32(self) -> int:
        return struct.unpack("<i", self.take(4))[0]

    def f32(self) -> float:
        return struct.unpack("<f", self.take(4))[0]

    def f64(self) -> float:
        return struct.unpack("<d", self.take(8))[0]

    def string(self) -> str:
        length = self.u32()
        if length == 0:
            return ""
        return self.take(length).decode("utf-8", errors="replace")


# Property type ids used by the binary format.
_T_STRING = 0x01
_T_BOOL = 0x02
_T_INT32 = 0x03
_T_FLOAT32 = 0x04
_T_FLOAT64 = 0x05
_T_UDIM = 0x06
_T_UDIM2 = 0x07
_T_RAY = 0x08
_T_FACES = 0x09
_T_AXES = 0x0A
_T_BRICKCOLOR = 0x0B
_T_COLOR3 = 0x0C
_T_VECTOR2 = 0x0D
_T_VECTOR3 = 0x0E
_T_VECTOR2INT16 = 0x0F
_T_CFRAME = 0x10
_T_QUATERNION = 0x11
_T_ENUM = 0x12
_T_REFERENT = 0x13
_T_VECTOR3INT16 = 0x14
_T_NUMBERSEQUENCE = 0x15
_T_COLORSEQUENCE = 0x16
_T_NUMBERRANGE = 0x17
_T_RECT = 0x18
_T_PHYSICALPROPERTIES = 0x19
_T_COLOR3UINT8 = 0x1A
_T_INTSEQUENCE = 0x1B
_T_SHAREDSTRING = 0x1C
_T_OPTIONALCFRAME = 0x1D
_T_FONT = 0x1E
_T_CONTENT = 0x1F
_T_SECURITYCAPABILITIES = 0x20
_T_ATTRIBUTES = 0x21


def _read_raw_property(reader: _Reader, type_id: int) -> bytes:
    """Consume one property value and return its raw bytes."""
    fixed_sizes = {
        _T_BOOL: 1,
        _T_INT32: 4,
        _T_FLOAT32: 4,
        _T_FLOAT64: 8,
        _T_UDIM: 8,
        _T_UDIM2: 16,
        _T_RAY: 24,
        _T_FACES: 1,
        _T_AXES: 1,
        _T_BRICKCOLOR: 4,
        _T_COLOR3: 12,
        _T_VECTOR2: 8,
        _T_VECTOR3: 12,
        _T_VECTOR2INT16: 4,
        _T_CFRAME: 48,
        _T_QUATERNION: 16,
        _T_ENUM: 4,
        _T_REFERENT: 4,
        _T_VECTOR3INT16: 6,
        _T_NUMBERRANGE: 8,
        _T_RECT: 16,
        _T_COLOR3UINT8: 3,
        _T_SHAREDSTRING: 4,
        _T_SECURITYCAPABILITIES: 8,
    }
    if type_id in fixed_sizes:
        return reader.take(fixed_sizes[type_id])
    if type_id in (_T_STRING, _T_CONTENT):
        start = reader.pos
        length = reader.u32()
        reader.take(length)
        return reader.data[start:reader.pos]
    if type_id == _T_OPTIONALCFRAME:
        start = reader.pos
        if reader.u8():
            reader.take(48)
        return reader.data[start:reader.pos]
    if type_id == _T_NUMBERSEQUENCE:
        start = reader.pos
        count = reader.u32()
        reader.take(12 * count)
        return reader.data[start:reader.pos]
    if type_id == _T_COLORSEQUENCE:
        start = reader.pos
        count = reader.u32()
        reader.take(20 * count)
        return reader.data[start:reader.pos]
    if type_id == _T_INTSEQUENCE:
        start = reader.pos
        count = reader.u32()
        reader.take(4 * count)
        return reader.data[start:reader.pos]
    if type_id == _T_PHYSICALPROPERTIES:
        start = reader.pos
        if reader.u8():
            reader.take(20)
        return reader.data[start:reader.pos]
    if type_id == _T_FONT:
        start = reader.pos
        length = reader.u32()
        reader.take(length)
        reader.take(4)  # weight (u16) + style (u8) + padding
        return reader.data[start:reader.pos]
    if type_id == _T_ATTRIBUTES:
        start = reader.pos
        count = reader.u32()
        for _ in range(count):
            name_len = reader.u32()
            reader.take(name_len)
            inner_type = reader.u8()
            _read_raw_property(reader, inner_type)
        return reader.data[start:reader.pos]
    raise RBXMError(f"unsupported property type 0x{type_id:02X}")


def _decode_property(type_id: int, raw: bytes):
    """Decode the handful of property types the exporter needs."""
    try:
        if type_id == _T_STRING or type_id == _T_CONTENT:
            length = struct.unpack_from("<I", raw, 0)[0]
            return raw[4:4 + length].decode("utf-8", errors="replace")
        if type_id == _T_BOOL:
            return bool(raw[0])
        if type_id == _T_INT32 or type_id == _T_ENUM or type_id == _T_REFERENT:
            return struct.unpack_from("<i", raw, 0)[0]
        if type_id == _T_FLOAT32:
            return struct.unpack_from("<f", raw, 0)[0]
        if type_id == _T_FLOAT64:
            return struct.unpack_from("<d", raw, 0)[0]
        if type_id == _T_COLOR3:
            return struct.unpack_from("<3f", raw, 0)
        if type_id == _T_VECTOR2:
            return struct.unpack_from("<2f", raw, 0)
        if type_id == _T_VECTOR3:
            return struct.unpack_from("<3f", raw, 0)
        if type_id == _T_CFRAME:
            return struct.unpack_from("<12f", raw, 0)
        if type_id == _T_UDIM:
            return struct.unpack_from("<fi", raw, 0)
        if type_id == _T_UDIM2:
            return struct.unpack_from("<fifi", raw, 0)
        if type_id == _T_BRICKCOLOR:
            return struct.unpack_from("<I", raw, 0)[0]
        if type_id == _T_COLOR3UINT8:
            return (raw[0], raw[1], raw[2])
    except (struct.error, IndexError):
        return None
    return None


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


@dataclass
class _ClassBlock:
    name: str
    referents: List[int] = field(default_factory=list)


def _parse_binary(data: bytes) -> Tuple[List[Instance], Optional[bytes], Optional[bytes]]:
    if not data.startswith(MAGIC):
        raise RBXMError("not a binary Roblox model file")
    reader = _Reader(data)
    reader.take(len(MAGIC) + len(SIGNATURE))
    reader.u16()  # format version
    reader.u32()  # class count
    reader.u32()  # instance count
    reader.take(8)  # reserved

    instances: Dict[int, Instance] = {}
    order: List[int] = []
    classes: List[_ClassBlock] = []
    parents: List[Tuple[int, int]] = []
    meta_raw: Optional[bytes] = None
    sstr_raw: Optional[bytes] = None

    while reader.remaining() >= 16:
        name = reader.take(4)
        compressed_len = reader.u32()
        uncompressed_len = reader.u32()
        reader.take(4)  # reserved
        payload = reader.take(compressed_len or uncompressed_len)
        if compressed_len:
            payload = _decompress_chunk(payload, uncompressed_len)
        chunk = _Reader(payload)

        if name == b"END\0":
            break
        if name == b"META":
            meta_raw = payload
        elif name == b"SSTR":
            sstr_raw = payload
        elif name == b"INST":
            class_index = chunk.u32()
            class_name = chunk.string()
            chunk.u8()  # is service
            count = chunk.u32()
            refs = [chunk.i32() for _ in range(count)]
            block = _ClassBlock(class_name, refs)
            if class_index < len(classes):
                classes[class_index] = block
            else:
                while len(classes) < class_index:
                    classes.append(_ClassBlock(""))
                classes.append(block)
            for ref in refs:
                inst = Instance(class_name=class_name, referent=ref)
                instances[ref] = inst
                order.append(ref)
                while True:
                    prop_name = chunk.string()
                    if not prop_name:
                        break
                    type_id = chunk.u8()
                    raw = _read_raw_property(chunk, type_id)
                    inst.props[prop_name] = (type_id, raw)
                    inst.values[prop_name] = _decode_property(type_id, raw)
        elif name == b"PROP":
            class_index = chunk.u32()
            prop_name = chunk.string()
            type_id = chunk.u8()
            if class_index < len(classes):
                for ref in classes[class_index].referents:
                    raw = _read_raw_property(chunk, type_id)
                    inst = instances.get(ref)
                    if inst is None:
                        continue
                    inst.props[prop_name] = (type_id, raw)
                    inst.values[prop_name] = _decode_property(type_id, raw)
        elif name == b"PRNT":
            chunk.u32()  # version
            count = chunk.u32()
            for _ in range(count):
                child = chunk.i32()
                parent = chunk.i32()
                parents.append((child, parent))

    roots: List[Instance] = []
    for child, parent in parents:
        inst = instances.get(child)
        if inst is None:
            continue
        if parent < 0 or parent not in instances:
            inst.parent = None
        else:
            inst.parent = instances[parent]
            inst.parent.children.append(inst)
    for ref in order:
        inst = instances.get(ref)
        if inst is not None and inst.parent is None:
            roots.append(inst)
    return roots, meta_raw, sstr_raw


def _parse_xml(data: bytes) -> List[Instance]:
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise RBXMError(f"invalid XML model: {exc}") from exc

    def parse_item(item) -> Instance:
        class_name = item.get("class", "Instance")
        inst = Instance(class_name=class_name)
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
                elif prop.tag == "Color3uint8":
                    try:
                        value = int(text)
                    except ValueError:
                        continue
                    inst.values[name] = (
                        (value >> 16) & 0xFF,
                        (value >> 8) & 0xFF,
                        value & 0xFF,
                    )
        for child in item.findall("Item"):
            parsed = parse_item(child)
            parsed.parent = inst
            inst.children.append(parsed)
        return inst

    roots: List[Instance] = []
    for item in root.findall("Item"):
        roots.append(parse_item(item))
    if not roots and root.tag == "Item":
        roots.append(parse_item(root))
    return roots


def read_model(path: str) -> List[Instance]:
    """Read a .rbxm/.rbxmx/.rbxl file and return its root instances."""
    with open(path, "rb") as handle:
        data = handle.read()
    if not data:
        raise RBXMError("file is empty")
    if data.startswith(MAGIC):
        roots, _, _ = _parse_binary(data)
        return roots
    return _parse_xml(data)


# ---------------------------------------------------------------------------
# Binary RBXM writer
# ---------------------------------------------------------------------------

def _pack_string(text: str) -> bytes:
    encoded = text.encode("utf-8")
    return struct.pack("<I", len(encoded)) + encoded


def _remap_referents(raw: bytes, mapping: Dict[int, int]) -> bytes:
    if len(raw) != 4:
        return raw
    old = struct.unpack("<i", raw)[0]
    new = mapping.get(old, -1)
    return struct.pack("<i", new)


def write_rbxm(roots: Iterable[Instance], path: str) -> int:
    """Write the given instances (and their children) as a binary RBXM file.

    Returns the number of instances written.
    """
    flat: List[Instance] = []

    def walk(inst: Instance) -> None:
        flat.append(inst)
        for child in inst.children:
            walk(child)

    for root in roots:
        walk(root)

    if not flat:
        raise RBXMError("nothing to write")

    mapping = {inst.referent: index for index, inst in enumerate(flat)}
    class_order: List[str] = []
    class_groups: Dict[str, List[Instance]] = {}
    for inst in flat:
        if inst.class_name not in class_groups:
            class_groups[inst.class_name] = []
            class_order.append(inst.class_name)
        class_groups[inst.class_name].append(inst)

    inst_chunks = bytearray()
    for class_name in class_order:
        group = class_groups[class_name]
        chunk = bytearray()
        chunk += struct.pack("<I", class_order.index(class_name))
        chunk += _pack_string(class_name)
        chunk += b"\x00"  # is service
        chunk += struct.pack("<I", len(group))
        for inst in group:
            chunk += struct.pack("<i", mapping[inst.referent])
        for inst in group:
            for prop_name, (type_id, raw) in inst.props.items():
                if type_id == _T_REFERENT:
                    raw = _remap_referents(raw, mapping)
                chunk += _pack_string(prop_name)
                chunk += bytes([type_id])
                chunk += raw
            chunk += b"\x00\x00\x00\x00"  # end of properties
        inst_chunks += _make_chunk(b"INST", bytes(chunk))

    prnt = bytearray()
    prnt += struct.pack("<I", 0)  # version
    prnt += struct.pack("<I", len(flat))
    for inst in flat:
        child_index = mapping[inst.referent]
        parent_index = mapping.get(inst.parent.referent, -1) if inst.parent is not None else -1
        prnt += struct.pack("<ii", child_index, parent_index)

    out = bytearray()
    out += MAGIC + SIGNATURE
    out += struct.pack("<H", 0)
    out += struct.pack("<I", len(class_order))
    out += struct.pack("<I", len(flat))
    out += b"\x00" * 8
    out += inst_chunks
    out += _make_chunk(b"PRNT", bytes(prnt))
    out += _make_chunk(b"END\0", b"")

    with open(path, "wb") as handle:
        handle.write(bytes(out))
    return len(flat)


def _make_chunk(name: bytes, payload: bytes) -> bytes:
    return name + struct.pack("<III", 0, len(payload), 0) + payload


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

# Enum.PartType values
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
    # Roblox cylinders are aligned along the X axis.
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
        (-x, -y, -z), (x, -y, -z),  # bottom back
        (-x, -y, z), (x, -y, z),    # bottom front
        (-x, y, -z), (x, y, -z),    # top back
    ]
    faces = [
        (0, 2, 3), (0, 3, 1),       # bottom
        (0, 1, 5), (0, 5, 4),       # back
        (4, 5, 3), (4, 3, 2),       # slope
        (0, 4, 2),                  # left
        (1, 3, 5),                  # right
    ]
    return verts, faces


def _transform(verts, rotation, position):
    x1, y1, z1, x2, y2, z2, x3, y3, z3 = rotation
    px, py, pz = position
    out = []
    for vx, vy, vz in verts:
        out.append((
            px + x1 * vx + x2 * vy + x3 * vz,
            py + y1 * vx + y2 * vy + y3 * vz,
            pz + z1 * vx + z2 * vy + z3 * vz,
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
            size = inst.get("Size")
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
        add(f'\t\tPolygonVertexIndex: *{len(polygon)} {{')
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


def export_model(
    source_path: str,
    output_dir: str,
    export_rbxm_file: bool = True,
    export_obj_file: bool = False,
    export_fbx_file: bool = False,
) -> Dict[str, object]:
    """Extract one model file into the requested formats."""
    roots = read_model(source_path)
    notes: List[str] = []
    meshes = collect_meshes(roots, notes)
    if not meshes:
        raise RBXMError("no parts with geometry were found")

    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(source_path))[0]
    written: List[str] = []
    triangles = 0

    if export_rbxm_file:
        target = os.path.join(output_dir, f"{stem}.rbxm")
        write_rbxm(roots, target)
        written.append(target)

    if export_obj_file:
        target = os.path.join(output_dir, f"{stem}.obj")
        triangles += export_obj(meshes, target)
        written.append(target)

    if export_fbx_file:
        target = os.path.join(output_dir, f"{stem}.fbx")
        if not triangles:
            triangles = sum(len(mesh.faces) for mesh in meshes)
        export_fbx(meshes, target)
        written.append(target)

    return {
        "source": source_path,
        "files": written,
        "parts": len(meshes),
        "triangles": triangles or sum(len(mesh.faces) for mesh in meshes),
        "notes": notes,
    }
