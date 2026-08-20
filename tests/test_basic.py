"""
tests/test_basic.py — Basic sanity tests for the capsnap package.
No network calls; all tests run offline using bundled assets.
"""

import base64
import os
import sys
import struct
import zlib

# ---------------------------------------------------------------------------
# Helpers: generate a tiny 1-channel PNG in memory (pure stdlib)
# ---------------------------------------------------------------------------

def _make_minimal_png(width: int = 20, height: int = 20) -> bytes:
    """Creates a valid 1×1-pixel greyscale PNG in memory."""
    def _chunk(tag: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + tag + data
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return c + struct.pack(">I", crc)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))

    raw = b""
    for y in range(height):
        raw += b"\x00"  # filter type None
        for x in range(width):
            # checkerboard pattern
            raw += b"\xff" if (x + y) % 2 == 0 else b"\x00"

    idat = _chunk(b"IDAT", zlib.compress(raw))
    iend = _chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


PNG_BYTES = _make_minimal_png()
PNG_B64 = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_import():
    from capsnap import OCR  # noqa
    assert OCR is not None


def test_read_bytes_returns_result():
    from capsnap import OCR
    ocr = OCR()
    result = ocr.read_bytes(PNG_BYTES)
    assert hasattr(result, "text")
    assert hasattr(result, "confidence")
    assert isinstance(result.text, str)
    assert isinstance(result.confidence, float)


def test_read_base64_returns_result():
    from capsnap import OCR
    ocr = OCR()
    result = ocr.read_base64(PNG_B64)
    assert hasattr(result, "text")
    assert 0.0 <= result.confidence <= 1.0


def test_captcha_mode_returns_result():
    from capsnap import OCR
    ocr = OCR(mode="captcha")
    result = ocr.read_bytes(PNG_BYTES)
    assert hasattr(result, "text")


def test_read_with_bytes():
    """read() should accept raw bytes directly."""
    from capsnap import OCR
    ocr = OCR()
    result = ocr.read(PNG_BYTES)
    assert isinstance(result.text, str)


def test_read_with_path(tmp_path):
    """read() should accept a file path string."""
    from capsnap import OCR
    img_path = tmp_path / "test.png"
    img_path.write_bytes(PNG_BYTES)
    ocr = OCR()
    result = ocr.read(str(img_path))
    assert isinstance(result.text, str)


def test_read_with_pathlib(tmp_path):
    """read() should accept a pathlib.Path object."""
    from capsnap import OCR
    import pathlib
    img_path = tmp_path / "test.png"
    img_path.write_bytes(PNG_BYTES)
    ocr = OCR()
    result = ocr.read(pathlib.Path(img_path))
    assert isinstance(result.text, str)


def test_read_with_base64_dataurl():
    """read() should accept a data-URL base64 string."""
    from capsnap import OCR
    ocr = OCR()
    result = ocr.read(PNG_B64)
    assert isinstance(result.text, str)

