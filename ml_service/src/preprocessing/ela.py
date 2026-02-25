# TODO: ELA generation code
import io
import numpy as np
from PIL import Image

def ela_rgb(pil_img: Image.Image, jpeg_quality: int = 90) -> np.ndarray:
    original = pil_img.convert("RGB")

    buf = io.BytesIO()
    original.save(buf, "JPEG", quality=jpeg_quality)
    buf.seek(0)
    compressed = Image.open(buf).convert("RGB")

    diff = np.abs(
        np.array(original, dtype=np.int16) - np.array(compressed, dtype=np.int16)
    ).astype(np.float32)

    mx = diff.max()
    if mx > 0:
        diff *= (255.0 / mx)

    return np.clip(diff, 0, 255).astype(np.uint8)