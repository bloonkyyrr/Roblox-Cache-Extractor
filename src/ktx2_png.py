"""KTX2 base-mip decoding and PNG writing for cache textures."""

import os
import struct
import zlib

try:
    import texture2ddecoder
except ImportError:
    texture2ddecoder = None

try:
    import zstandard
except ImportError:
    zstandard = None

try:
    from compression import zstd as builtin_zstd
except ImportError:
    builtin_zstd = None


_KTX2_IDENTIFIER = b"\xabKTX 20\xbb\r\n\x1a\n"
_MAX_IMAGE_PIXELS = 64 * 1024 * 1024
_MAX_LEVEL_BYTES = 512 * 1024 * 1024


def _decode_bc1_bgra(data, width, height, force_opaque=False):
    pixels = bytearray(width * height * 4)
    blocks_x = (width + 3) // 4
    blocks_y = (height + 3) // 4
    source_offset = 0

    def expand_565(color):
        red = (color >> 11) & 0x1f
        green = (color >> 5) & 0x3f
        blue = color & 0x1f
        return ((red << 3) | (red >> 2),
                (green << 2) | (green >> 4),
                (blue << 3) | (blue >> 2))

    for block_y in range(blocks_y):
        for block_x in range(blocks_x):
            color0 = int.from_bytes(data[source_offset:source_offset + 2], "little")
            color1 = int.from_bytes(data[source_offset + 2:source_offset + 4], "little")
            indices = int.from_bytes(data[source_offset + 4:source_offset + 8], "little")
            source_offset += 8
            rgb0 = expand_565(color0)
            rgb1 = expand_565(color1)

            if color0 > color1 or force_opaque:
                palette = (
                    (*rgb0, 255),
                    (*rgb1, 255),
                    tuple((2 * rgb0[channel] + rgb1[channel]) // 3 for channel in range(3)) + (255,),
                    tuple((rgb0[channel] + 2 * rgb1[channel]) // 3 for channel in range(3)) + (255,),
                )
            else:
                palette = (
                    (*rgb0, 255),
                    (*rgb1, 255),
                    tuple((rgb0[channel] + rgb1[channel]) // 2 for channel in range(3)) + (255,),
                    (0, 0, 0, 0),
                )

            for local_y in range(4):
                pixel_y = block_y * 4 + local_y
                if pixel_y >= height:
                    continue
                for local_x in range(4):
                    pixel_x = block_x * 4 + local_x
                    if pixel_x >= width:
                        continue
                    pixel_index = local_y * 4 + local_x
                    red, green, blue, alpha = palette[(indices >> (pixel_index * 2)) & 0x03]
                    target_offset = (pixel_y * width + pixel_x) * 4
                    pixels[target_offset:target_offset + 4] = bytes((blue, green, red, alpha))
    return bytes(pixels)


def _decode_bc3_bgra(data, width, height):
    pixels = bytearray(width * height * 4)
    blocks_x = (width + 3) // 4
    blocks_y = (height + 3) // 4
    source_offset = 0

    for block_y in range(blocks_y):
        for block_x in range(blocks_x):
            alpha0 = data[source_offset]
            alpha1 = data[source_offset + 1]
            alpha_indices = int.from_bytes(data[source_offset + 2:source_offset + 8], "little")
            color_data = data[source_offset + 8:source_offset + 16]
            source_offset += 16

            if alpha0 > alpha1:
                alpha_palette = [alpha0, alpha1] + [
                    ((8 - index) * alpha0 + (index - 1) * alpha1) // 7
                    for index in range(2, 8)
                ]
            else:
                alpha_palette = [
                    alpha0, alpha1,
                    (4 * alpha0 + alpha1) // 5,
                    (3 * alpha0 + 2 * alpha1) // 5,
                    (2 * alpha0 + 3 * alpha1) // 5,
                    (alpha0 + 4 * alpha1) // 5,
                    0, 255,
                ]

            color_pixels = _decode_bc1_bgra(color_data, 4, 4, force_opaque=True)
            for local_y in range(4):
                pixel_y = block_y * 4 + local_y
                if pixel_y >= height:
                    continue
                for local_x in range(4):
                    pixel_x = block_x * 4 + local_x
                    if pixel_x >= width:
                        continue
                    pixel_index = local_y * 4 + local_x
                    alpha = alpha_palette[(alpha_indices >> (pixel_index * 3)) & 0x07]
                    source_pixel_offset = pixel_index * 4
                    target_offset = (pixel_y * width + pixel_x) * 4
                    pixels[target_offset:target_offset + 4] = (
                        color_pixels[source_pixel_offset:source_pixel_offset + 3] + bytes((alpha,))
                    )
    return bytes(pixels)


def _write_png(path, width, height, pixels, color_type, bgra=False):
    """Write scanlines as a PNG without needing Pillow or Qt."""
    channels = {0: 1, 2: 3, 6: 4}[color_type]
    row_bytes = width * channels
    if len(pixels) < row_bytes * height:
        raise ValueError("decoded pixel data is truncated")
    if bgra:
        rgba = bytearray(pixels[:row_bytes * height])
        swap_chunk_size = 4 * 1024 * 1024
        for chunk_start in range(0, len(rgba), swap_chunk_size):
            chunk_end = min(len(rgba), chunk_start + swap_chunk_size)
            chunk = rgba[chunk_start:chunk_end]
            red = chunk[2::4]
            blue = chunk[0::4]
            chunk[0::4] = red
            chunk[2::4] = blue
            rgba[chunk_start:chunk_end] = chunk
        pixels = rgba

    temporary_path = path + ".writing"

    def write_chunk(output, chunk_type, payload):
        output.write(struct.pack(">I", len(payload)))
        output.write(chunk_type)
        output.write(payload)
        output.write(struct.pack(">I", zlib.crc32(chunk_type + payload) & 0xffffffff))

    try:
        with open(temporary_path, "wb") as output:
            output.write(b"\x89PNG\r\n\x1a\n")
            write_chunk(output, b"IHDR", struct.pack(">2I5B", width, height, 8, color_type, 0, 0, 0))
            compressor = zlib.compressobj(level=6)
            pending = bytearray()
            for row_index in range(height):
                start = row_index * row_bytes
                row = pixels[start:start + row_bytes]
                pending.extend(compressor.compress(b"\x00" + row))
                if len(pending) >= 1024 * 1024:
                    write_chunk(output, b"IDAT", bytes(pending))
                    pending.clear()
            pending.extend(compressor.flush())
            if pending:
                write_chunk(output, b"IDAT", bytes(pending))
            write_chunk(output, b"IEND", b"")
        os.replace(temporary_path, path)
    except Exception:
        try:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)
        except OSError:
            pass
        raise


def _decompress_zstd(data, expected_size):
    if not expected_size or expected_size > _MAX_LEVEL_BYTES:
        raise ValueError("KTX2 decompressed size is invalid or too large")
    if zstandard is not None:
        return zstandard.ZstdDecompressor().decompress(
            data, max_output_size=expected_size
        )
    if builtin_zstd is not None:
        decompressor = builtin_zstd.ZstdDecompressor()
        decoded = decompressor.decompress(data, max_length=expected_size + 1)
        while not decompressor.eof and not decompressor.needs_input:
            remaining = expected_size + 1 - len(decoded)
            if remaining <= 0:
                raise ValueError("KTX2 decompressed data exceeds its declared size")
            decoded += decompressor.decompress(b"", max_length=remaining)
        if not decompressor.eof:
            raise ValueError("KTX2 Zstandard stream is incomplete")
        return decoded
    raise ValueError("Zstandard support is missing; run install_dependencies.bat")


def convert_ktx2_to_png(input_path, output_path):
    """Decode a supported KTX2 base mip and save it as PNG. Returns (ok, error)."""
    try:
        file_size = os.path.getsize(input_path)
        with open(input_path, "rb") as texture_file:
            header = texture_file.read(80)
            if len(header) < 80 or header[:12] != _KTX2_IDENTIFIER:
                return False, "invalid KTX2 header"

            fields = struct.unpack_from("<13I2Q", header, 12)
            (vk_format, _type_size, width, height, depth, _layer_count, face_count,
             level_count, supercompression, _dfd_offset, _dfd_length, _kvd_offset,
             _kvd_length, _sgd_offset, _sgd_length) = fields
            if not width or not height or width > 32768 or height > 32768:
                return False, "unsupported or invalid image dimensions"
            if width * height > _MAX_IMAGE_PIXELS:
                return False, "KTX2 image is too large to decode safely"
            if depth > 1 or face_count not in (1, 6):
                return False, "only 2D KTX2 images are supported"
            level_count = max(1, level_count)
            if level_count > 32 or 80 + level_count * 24 > file_size:
                return False, "KTX2 mip level index is invalid"

            texture_file.seek(80)
            level_index = texture_file.read(24)
            if len(level_index) != 24:
                return False, "KTX2 mip level index is incomplete"
            level_offset, level_length, uncompressed_length = struct.unpack("<QQQ", level_index)
            if (level_offset < 80 + level_count * 24 or not level_length
                    or level_length > _MAX_LEVEL_BYTES or level_offset + level_length > file_size):
                return False, "KTX2 base mip level has an invalid size"
            texture_file.seek(level_offset)
            image_data = texture_file.read(level_length)
            if len(image_data) != level_length:
                return False, "KTX2 base mip level is truncated"

        if supercompression == 2:
            image_data = _decompress_zstd(image_data, uncompressed_length)
        elif supercompression == 3:
            if not uncompressed_length or uncompressed_length > _MAX_LEVEL_BYTES:
                return False, "KTX2 decompressed size is invalid or too large"
            decompressor = zlib.decompressobj()
            image_data = decompressor.decompress(image_data, uncompressed_length + 1)
            if len(image_data) > uncompressed_length or decompressor.unconsumed_tail:
                return False, "KTX2 decompressed data exceeds its declared size"
            image_data += decompressor.flush()
        elif supercompression == 1:
            return False, "BasisLZ textures need a Basis Universal transcoder"
        elif supercompression != 0:
            return False, "unsupported KTX2 supercompression scheme"
        if uncompressed_length and len(image_data) != uncompressed_length:
            return False, "KTX2 base mip size does not match its header"

        if vk_format in (37, 43):  # RGBA8 UNORM / SRGB
            image_size = width * height * 4
            if len(image_data) < image_size:
                return False, "KTX2 RGBA pixel data is truncated"
            pixels, color_type, bgra = image_data[:image_size], 6, False
        elif vk_format in (23, 29):  # RGB8 UNORM / SRGB
            image_size = width * height * 3
            if len(image_data) < image_size:
                return False, "KTX2 RGB pixel data is truncated"
            pixels, color_type, bgra = image_data[:image_size], 2, False
        elif vk_format == 9:  # R8 UNORM
            image_size = width * height
            if len(image_data) < image_size:
                return False, "KTX2 grayscale pixel data is truncated"
            pixels, color_type, bgra = image_data[:image_size], 0, False
        else:
            decoder = None
            block_width = block_height = 4
            bytes_per_block = 0
            pure_bc1 = pure_bc3 = is_astc = False
            if vk_format in (131, 132, 133, 134):
                bytes_per_block = 8
                decoder = getattr(texture2ddecoder, "decode_bc1", None) if texture2ddecoder else None
                pure_bc1 = decoder is None
            elif vk_format in (137, 138):
                bytes_per_block = 16
                decoder = getattr(texture2ddecoder, "decode_bc3", None) if texture2ddecoder else None
                pure_bc3 = decoder is None
            else:
                codec_formats = {
                    139: ("decode_bc4", 8), 141: ("decode_bc5", 16),
                    145: ("decode_bc7", 16), 146: ("decode_bc7", 16),
                    147: ("decode_etc2", 8), 148: ("decode_etc2", 8),
                    149: ("decode_etc2a1", 8), 150: ("decode_etc2a1", 8),
                    151: ("decode_etc2a8", 16), 152: ("decode_etc2a8", 16),
                }
                codec = codec_formats.get(vk_format)
                if codec and texture2ddecoder is not None:
                    decoder = getattr(texture2ddecoder, codec[0], None)
                    bytes_per_block = codec[1]
                astc_block_sizes = {
                    157: (4, 4), 158: (4, 4), 159: (5, 4), 160: (5, 4),
                    161: (5, 5), 162: (5, 5), 163: (6, 5), 164: (6, 5),
                    165: (6, 6), 166: (6, 6), 167: (8, 5), 168: (8, 5),
                    169: (8, 6), 170: (8, 6), 171: (8, 8), 172: (8, 8),
                    173: (10, 5), 174: (10, 5), 175: (10, 6), 176: (10, 6),
                    177: (10, 8), 178: (10, 8), 179: (10, 10), 180: (10, 10),
                    181: (12, 10), 182: (12, 10), 183: (12, 12), 184: (12, 12),
                }
                if vk_format in astc_block_sizes and texture2ddecoder is not None:
                    decoder = getattr(texture2ddecoder, "decode_astc", None)
                    block_width, block_height = astc_block_sizes[vk_format]
                    bytes_per_block = 16
                    is_astc = True

            if not (decoder or pure_bc1 or pure_bc3):
                return False, "this KTX2 pixel format is not supported; install requirements or KTX CLI"
            blocks_x = (width + block_width - 1) // block_width
            blocks_y = (height + block_height - 1) // block_height
            image_size = blocks_x * blocks_y * bytes_per_block
            if len(image_data) < image_size:
                return False, "KTX2 compressed pixel data is truncated"
            if pure_bc1:
                pixels = _decode_bc1_bgra(
                    image_data[:image_size], width, height,
                    force_opaque=vk_format in (131, 132),
                )
            elif pure_bc3:
                pixels = _decode_bc3_bgra(image_data[:image_size], width, height)
            elif is_astc:
                pixels = decoder(image_data[:image_size], width, height, block_width, block_height)
            else:
                pixels = decoder(image_data[:image_size], width, height)
            if len(pixels) != width * height * 4:
                return False, "texture decoder returned an invalid pixel buffer"
            color_type, bgra = 6, True

        _write_png(output_path, width, height, pixels, color_type, bgra=bgra)
        return True, ""
    except Exception as error:
        return False, str(error)
