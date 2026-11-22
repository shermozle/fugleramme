"""The panel page packed for an external e-ink panel, so a microcontroller
can stream it to the glass without touching it.

A 20-byte little-endian header, then the payload:

    0  magic "E6F1" (u32)   4  version (u16)       6  encoding (u8)
    7  poll minutes (u8)    8  width (u16)         10 height (u16)
    12 payload length (u32) 16 payload CRC-32 (u32)

Encoding 1 is the controllers' own colour codes, two pixels a byte with the left
one in the high nibble, in the order the glass's controllers read them. The
13.3" glass is driven as two halves: every row's left half goes to the first
controller (CS0), then every row's right half to the second (CS1), so the
payload is those two runs back to back. The 7.3" glass has one controller,
which reads its landscape rows top to bottom. The 7.5" BigMe F7 Pro has two
512x600 controllers; each row's left 256 bytes go to one and its right 256
bytes to the other, so its wire stream is also ordinary row-major 4bpp data.

The poll is the frame's Panel refresh in minutes, 0 meaning as soon as it
changes: the client sleeps that long between fetches, its own minimum on top.
"""

from __future__ import annotations

import struct
import zlib

import numpy as np
from PIL import Image

from .config import PANEL_SIZES
from .panel import to_glass

MAGIC = b"E6F1"
VERSION = 1
ENCODING = 1
HEADER = struct.Struct("<4sHBBHHII")

# The 13.3" Seeed panel needs a special scan order. The 7.3" and the BigMe
# F7 Pro's paired controllers both consume the row-major stream directly.
_SEEED_13_3 = PANEL_SIZES["13.3"]
_F7_PRO = PANEL_SIZES["7.5"]
GLASSES = (PANEL_SIZES["7.3"], _F7_PRO, _SEEED_13_3)

# `dither`'s palette order - black, white, yellow, red, blue, green - to the
# controllers' codes, which skip 4.
_NATIVE = np.array([0, 1, 2, 3, 5, 6], dtype=np.uint8)


def native_frame(frame: Image.Image, rotation: int) -> Image.Image:
    """The external panel's native orientation, before any controller split."""
    glass = to_glass(frame, rotation)
    # The F7 Pro's memory origin is the opposite corner from the render canvas.
    return glass.transpose(Image.Transpose.ROTATE_180) if glass.size == _F7_PRO else glass


def encode(frame: Image.Image, rotation: int, poll: int) -> bytes:
    """The dithered panel page, as it hangs, packed for the glass."""
    glass = native_frame(frame, rotation)
    halves = glass.size == _SEEED_13_3
    # The 13.3"'s controllers read it a quarter turn clockwise from the landscape
    # glass, as Pimoroni's driver turns it.
    scan = glass.transpose(Image.Transpose.ROTATE_270) if halves else glass
    pixels = np.asarray(scan, dtype=np.uint8)
    height, width = pixels.shape
    codes = _NATIVE[pixels]
    ordered = np.concatenate(np.hsplit(codes, 2)) if halves else codes
    payload = ((ordered[:, 0::2] << 4) | ordered[:, 1::2]).tobytes()
    header = HEADER.pack(
        MAGIC,
        VERSION,
        ENCODING,
        min(poll, 255),
        width,
        height,
        len(payload),
        zlib.crc32(payload),
    )
    return header + payload
