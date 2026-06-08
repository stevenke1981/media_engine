import struct, zlib

w, h = 64, 48
raw = b"".join(
    bytes([(x + y) % 256, (x * 2) % 256, (y * 3) % 256, 255])
    for y in range(h)
    for x in range(w)
)


def chunk(typ, data):
    c = typ.encode() + data
    crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
    return struct.pack(">I", len(data)) + c + crc


sig = b"\x89PNG\r\n\x1a\n"
ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
idat = zlib.compress(raw)
png = sig + chunk("IHDR", ihdr) + chunk("IDAT", idat) + chunk("IEND", b"")

with open(r"D:\rust-engi\media_engine\test_pipeline_input.png", "wb") as f:
    f.write(png)
print(f"Created {w}x{h} test PNG -> test_pipeline_input.png ({len(png)} bytes)")
