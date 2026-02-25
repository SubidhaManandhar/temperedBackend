# TODO: noise residual map generation
import numpy as np
import cv2
from .srm_filter_bank import get_srm30_kernels

def srm30_residual_rgb(rgb_uint8: np.ndarray) -> np.ndarray:
    """
    Input : RGB uint8 (H,W,3)
    Output: RGB uint8 residual map (H,W,3) made by grouping 30 filter responses into 3 channels.
    """

    gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    kernels = get_srm30_kernels()

    responses = []
    for ker in kernels:
        r = cv2.filter2D(gray, ddepth=cv2.CV_32F, kernel=ker, borderType=cv2.BORDER_REFLECT)
        responses.append(np.abs(r))

    vol = np.stack(responses, axis=-1)  # (H,W,30)

    # Group filters into 3 groups -> 3 channels (mean abs response per group)
    g1 = np.mean(vol[:, :, 0:10], axis=-1)
    g2 = np.mean(vol[:, :, 10:20], axis=-1)
    g3 = np.mean(vol[:, :, 20:30], axis=-1)
    out = np.stack([g1, g2, g3], axis=-1)  # (H,W,3)

    # Robust per-channel normalization (percentile)
    out_norm = np.zeros_like(out, dtype=np.float32)
    for c in range(3):
        ch = out[:, :, c]
        lo = np.percentile(ch, 1)
        hi = np.percentile(ch, 99)
        if hi - lo < 1e-6:
            hi = lo + 1.0
        chn = (ch - lo) / (hi - lo)
        out_norm[:, :, c] = np.clip(chn, 0, 1)

    return (out_norm * 255.0).astype(np.uint8)