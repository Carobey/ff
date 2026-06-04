"""QR-code decoder for Fiscal receipt QR strings.

Sberbank receipts carry a QR code whose payload is a URL-encoded fiscal string:
  t=<datetime>&s=<amount>&fn=<FN>&i=<FD>&fp=<FP>&n=1

We decode the image with pyzbar (libzbar) and fall back to nothing — the caller
is responsible for asking the user to re-shoot on failure.
"""

from __future__ import annotations

import re
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from pyzbar import pyzbar

# Minimal validation: Russian fiscal QR must contain fn= and fp=
_FISCAL_RE = re.compile(r"fn=\d+.*fp=\d+", re.IGNORECASE)


def decode_qr(image_bytes: bytes) -> str | None:
    """Decode first QR code found in *image_bytes*.

    Returns the decoded string if it looks like a fiscal QR, otherwise None.
    """
    try:
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        np_image = np.array(pil_image)
        # OpenCV uses BGR, PIL gives RGB — convert
        bgr = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    except Exception:
        return None

    decoded = pyzbar.decode(bgr)
    for obj in decoded:
        try:
            text: str = obj.data.decode("utf-8")
        except (UnicodeDecodeError, AttributeError):
            continue
        if _FISCAL_RE.search(text):
            return text

    # Try once more with grayscale + sharpened image if direct decode failed
    try:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        sharp = cv2.filter2D(
            gray,
            ddepth=-1,
            kernel=np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]),
        )
        decoded2 = pyzbar.decode(sharp)
        for obj in decoded2:
            try:
                text2: str = obj.data.decode("utf-8")
            except (UnicodeDecodeError, AttributeError):
                continue
            if _FISCAL_RE.search(text2):
                return text2
    except Exception:  # noqa: S110
        pass

    return None


def compose_fiscal_qr(fields: dict[str, str]) -> str:
    """Build a synthetic fiscal QR payload from extracted fields.

    Used when the QR image can't be decoded directly but a vision LLM has
    recovered the fiscal fields from the receipt text. The output mirrors
    what ``decode_qr`` would normally return.
    """
    return (
        f"t={fields.get('t', '')}&s={fields.get('s', '0')}"
        f"&fn={fields.get('fn', '')}&i={fields.get('i', '')}"
        f"&fp={fields.get('fp', '')}&n=1"
    )


def parse_fiscal_qr(qr_string: str) -> dict[str, str]:
    """Parse key=value pairs from fiscal QR string into a plain dict.

    Example input:
      t=20260502T123456&s=1234.56&fn=9289000100123456&i=12345&fp=1234567890&n=1

    Returns e.g.::
        {'t': '20260502T123456', 's': '1234.56', 'fn': '...', 'i': '...', 'fp': '...', 'n': '1'}
    """
    result: dict[str, str] = {}
    # Some QR payloads start with a URL prefix
    if "?" in qr_string:
        qr_string = qr_string.split("?", 1)[1]
    for pair in qr_string.split("&"):
        if "=" in pair:
            k, v = pair.split("=", 1)
            result[k.strip().lower()] = v.strip()
    return result
