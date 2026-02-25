import numpy as np

def get_srm30_kernels():
    k = []

    # --- 3x3 laplacian / edge / second-derivative family ---
    k += [
        np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], np.float32),
        np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], np.float32),
        np.array([[1,-2,1],[-2,4,-2],[1,-2,1]], np.float32),
        np.array([[0,1,0],[1,-4,1],[0,1,0]], np.float32),
    ]

    # --- 3x3 sobel/prewitt style (high-pass) ---
    k += [
        np.array([[-1,0,1],[-2,0,2],[-1,0,1]], np.float32),
        np.array([[-1,-2,-1],[0,0,0],[1,2,1]], np.float32),
        np.array([[-1,0,1],[-1,0,1],[-1,0,1]], np.float32),
        np.array([[-1,-1,-1],[0,0,0],[1,1,1]], np.float32),
        np.array([[1,1,1],[0,0,0],[-1,-1,-1]], np.float32),
        np.array([[1,0,-1],[1,0,-1],[1,0,-1]], np.float32),
    ]

    # --- 5x5 SRM-like center-surround / high-pass ---
    k += [
        np.array([[0,0,0,0,0],
                  [0,-1,2,-1,0],
                  [0,2,-4,2,0],
                  [0,-1,2,-1,0],
                  [0,0,0,0,0]], np.float32),

        np.array([[-1,2,-2,2,-1],
                  [2,-6,8,-6,2],
                  [-2,8,-12,8,-2],
                  [2,-6,8,-6,2],
                  [-1,2,-2,2,-1]], np.float32),

        np.array([[0,0,0,0,0],
                  [0,0,-1,0,0],
                  [0,-1,4,-1,0],
                  [0,0,-1,0,0],
                  [0,0,0,0,0]], np.float32),
    ]

    # --- 5x5 directional high-pass variants ---
    k += [
        np.array([[0,0,0,0,0],
                  [0,0,0,0,0],
                  [-1,2,-2,2,-1],
                  [0,0,0,0,0],
                  [0,0,0,0,0]], np.float32),

        np.array([[0,0,-1,0,0],
                  [0,0,2,0,0],
                  [0,0,-2,0,0],
                  [0,0,2,0,0],
                  [0,0,-1,0,0]], np.float32),

        np.array([[-1,0,0,0,0],
                  [0,2,0,0,0],
                  [0,0,-2,0,0],
                  [0,0,0,2,0],
                  [0,0,0,0,-1]], np.float32),

        np.array([[0,0,0,0,-1],
                  [0,0,0,2,0],
                  [0,0,-2,0,0],
                  [0,2,0,0,0],
                  [-1,0,0,0,0]], np.float32),
    ]

    # --- 3x3 “compact SRM-like” residual patterns ---
    k += [
        np.array([[0,0,0],[0,1,-1],[0,-1,1]], np.float32),
        np.array([[0,0,0],[1,-1,0],[-1,1,0]], np.float32),
        np.array([[1,-1,0],[-1,1,0],[0,0,0]], np.float32),
        np.array([[0,-1,1],[0,1,-1],[0,0,0]], np.float32),
        np.array([[0,0,0],[0,-1,1],[0,1,-1]], np.float32),
        np.array([[0,0,0],[-1,1,0],[1,-1,0]], np.float32),
    ]

    # --- 5x5 more high-pass / checker / ripple patterns ---
    k += [
        np.array([[0,0,0,0,0],
                  [0,-1,0,1,0],
                  [0,0,0,0,0],
                  [0,1,0,-1,0],
                  [0,0,0,0,0]], np.float32),

        np.array([[0,0,0,0,0],
                  [0,1,-2,1,0],
                  [0,-2,4,-2,0],
                  [0,1,-2,1,0],
                  [0,0,0,0,0]], np.float32),

        np.array([[-1,0,2,0,-1],
                  [0,0,0,0,0],
                  [2,0,-4,0,2],
                  [0,0,0,0,0],
                  [-1,0,2,0,-1]], np.float32),

        np.array([[1,-2,2,-2,1],
                  [-2,4,-4,4,-2],
                  [2,-4,4,-4,2],
                  [-2,4,-4,4,-2],
                  [1,-2,2,-2,1]], np.float32),
    ]

    # --- pad with additional slightly perturbed variants to reach 30 ---
    # (small rotations / sign flips of earlier kernels)
    base = k.copy()
    while len(k) < 30:
        idx = len(k) % len(base)
        ker = base[idx]
        ker2 = np.rot90(ker, 1)
        k.append(ker2)

    # Normalize kernels to keep response magnitudes stable
    normed = []
    for ker in k[:30]:
        denom = np.sum(np.abs(ker))
        normed.append(ker / denom if denom > 0 else ker)
    return normed