import os
import uuid
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

from src.preprocessing.ela import ela_rgb
from src.preprocessing.noise_map import srm30_residual_rgb

IMG_SIZE = (224, 224)
JPEG_QUALITY = 90

BINARY_CLASSES = ["authentic", "tampered"]
TAMPER_CLASSES = ["copy_move", "enhancement", "removal_inpainting", "splicing"]

CANNY1 = 60
CANNY2 = 140
EDGE_THICKNESS = 1

MIN_HW = 14
TOP_K_LAYERS = 12
KEEP_PERCENT = 10
MIN_COMPONENT_AREA = 80

USE_BBOX = True
DRAW_CONTOUR = True
OVERLAY_ALPHA = 0.55


def preprocess_image(image_path):
    pil = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    rgb_uint8 = np.array(pil, dtype=np.uint8)

    ela_uint8 = ela_rgb(pil, JPEG_QUALITY)

    try:
        noise_uint8 = srm30_residual_rgb(pil)
    except Exception:
        noise_uint8 = srm30_residual_rgb(rgb_uint8)

    ela = (ela_uint8.astype(np.float32) / 255.0)[None, ...]
    noise = (noise_uint8.astype(np.float32) / 255.0)[None, ...]

    return pil, rgb_uint8, ela, noise, ela_uint8, noise_uint8


def list_candidate_conv_layers(model: tf.keras.Model, min_hw=14):
    candidates = []

    def walk(m):
        for layer in m.layers:
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                try:
                    sh = layer.output_shape
                    if isinstance(sh, (tuple, list)) and len(sh) >= 4:
                        h, w = sh[1], sh[2]
                        if h is not None and w is not None and h >= min_hw and w >= min_hw:
                            candidates.append(layer.name)
                except Exception:
                    pass
            if isinstance(layer, tf.keras.Model):
                walk(layer)

    walk(model)

    if not candidates:
        def walk_all(m):
            for layer in m.layers:
                if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                    candidates.append(layer.name)
                if isinstance(layer, tf.keras.Model):
                    walk_all(layer)
        candidates.clear()
        walk_all(model)

    seen = set()
    uniq = []
    for n in candidates:
        if n not in seen:
            uniq.append(n)
            seen.add(n)
    return uniq


def predict_probs(model, ela, noise):
    return model({"ela": ela, "noise": noise}, training=False).numpy()[0]


def make_gradcam_heatmap_for_layer(model, ela_input, noise_input, class_index, conv_layer_name):
    conv_layer = model.get_layer(conv_layer_name)
    grad_model = tf.keras.Model(model.inputs, [conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model({"ela": ela_input, "noise": noise_input}, training=False)
        score = preds[:, class_index]

    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(conv_out[0] * weights, axis=-1)

    cam = tf.maximum(cam, 0)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy()


def cam_to_binary_mask(cam01, keep_percent=10, out_hw=(224, 224)):
    cam_r = cv2.resize(cam01, out_hw)
    cam_r = np.clip(cam_r, 0, 1)

    thr = np.percentile(cam_r, 100 - keep_percent)
    mask = (cam_r >= thr).astype(np.uint8) * 255

    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return cam_r, mask


def keep_largest_component(mask255, min_area=80):
    contours, _ = cv2.findContours(mask255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask255), None

    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not contours:
        return np.zeros_like(mask255), None

    largest = max(contours, key=cv2.contourArea)
    clean = np.zeros_like(mask255)
    cv2.drawContours(clean, [largest], -1, 255, thickness=-1)
    return clean, largest


def apply_mask_on_inputs(ela, noise, mask255, mode="zero"):
    mask = (mask255.astype(np.float32) / 255.0)[None, ..., None]
    if mode == "zero":
        return (ela * (1.0 - mask)).astype(np.float32), (noise * (1.0 - mask)).astype(np.float32)

    ela_np = ela[0].copy()
    noise_np = noise[0].copy()
    ela_blur = cv2.GaussianBlur(ela_np, (11, 11), 0)
    noise_blur = cv2.GaussianBlur(noise_np, (11, 11), 0)
    m2 = mask[0]
    ela_np = ela_np * (1 - m2) + ela_blur * m2
    noise_np = noise_np * (1 - m2) + noise_blur * m2
    return ela_np[None, ...].astype(np.float32), noise_np[None, ...].astype(np.float32)


def auto_select_best_layer(model, ela, noise, target_class, min_hw=14, keep_percent=10, top_k=12):
    base_probs = predict_probs(model, ela, noise)
    base = float(base_probs[target_class])

    layers = list_candidate_conv_layers(model, min_hw=min_hw)
    layers_to_test = layers[-top_k:] if len(layers) > top_k else layers

    best = None
    for lname in layers_to_test:
        cam = make_gradcam_heatmap_for_layer(model, ela, noise, target_class, lname)
        cam01, mask = cam_to_binary_mask(cam, keep_percent=keep_percent, out_hw=IMG_SIZE)

        ela_m, noise_m = apply_mask_on_inputs(ela, noise, mask, mode="zero")
        masked_probs = predict_probs(model, ela_m, noise_m)
        masked = float(masked_probs[target_class])

        drop = base - masked
        area = float(mask.mean() / 255.0)
        score = drop - 0.10 * area

        if best is None or score > best[1]:
            best = (lname, score, cam01, mask)

    return best  # (best_layer, score, cam01, raw_mask)


def overlay_suspected(original_rgb, cam01, mask255, alpha=0.55):
    H, W = IMG_SIZE
    orig = cv2.resize(original_rgb, (W, H))

    heat_u8 = np.uint8(255 * cam01)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

    mask3 = cv2.cvtColor(mask255, cv2.COLOR_GRAY2BGR)
    heat_color = cv2.bitwise_and(heat_color, mask3)

    orig_bgr = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)
    out = cv2.addWeighted(orig_bgr, 1 - alpha, heat_color, alpha, 0)

    gray = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, CANNY1, CANNY2)
    if EDGE_THICKNESS > 1:
        k = np.ones((EDGE_THICKNESS, EDGE_THICKNESS), np.uint8)
        edges = cv2.dilate(edges, k, iterations=1)
    out[edges > 0] = (0, 0, 0)

    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def draw_bbox(rgb, contour):
    if contour is None:
        return rgb, None
    x, y, w, h = cv2.boundingRect(contour)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.rectangle(bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}


def draw_contour(rgb, contour):
    if contour is None:
        return rgb
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.drawContours(bgr, [contour], -1, (0, 255, 0), 2)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def predict_with_gradcam(image_path: str, stageA: tf.keras.Model, stageB: tf.keras.Model, outputs_dir: str):
    pil, _, ela, noise, _, _ = preprocess_image(image_path)

    predA = stageA({"ela": ela, "noise": noise}, training=False).numpy()[0]
    idxA = int(np.argmax(predA))
    confA = float(predA[idxA]) * 100.0
    stageA_label = BINARY_CLASSES[idxA]

    if idxA == 0:
        return {
            "stageA_label": stageA_label,
            "stageA_confidence": confA,
            "label": "authentic",
            "confidence": confA,
            "heatmap_file": None,
            "mask_file": None,
            "bbox": None,
            "best_layer": None
        }

    predB = stageB({"ela": ela, "noise": noise}, training=False).numpy()[0]
    idxB = int(np.argmax(predB))
    confB = float(predB[idxB]) * 100.0
    final_label = TAMPER_CLASSES[idxB]

    best_layer, _, cam01, raw_mask = auto_select_best_layer(
        stageB, ela, noise, idxB, min_hw=MIN_HW, keep_percent=KEEP_PERCENT, top_k=TOP_K_LAYERS
    )

    mask, contour = keep_largest_component(raw_mask, min_area=MIN_COMPONENT_AREA)

    original = np.array(pil)
    overlay_img = overlay_suspected(original, cam01, mask, alpha=OVERLAY_ALPHA)

    annotated = overlay_img
    if DRAW_CONTOUR:
        annotated = draw_contour(annotated, contour)

    bbox = None
    if USE_BBOX:
        annotated, bbox = draw_bbox(annotated, contour)

    heatmaps_dir = os.path.join(outputs_dir, "heatmaps")
    masks_dir = os.path.join(outputs_dir, "masks")
    os.makedirs(heatmaps_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    file_id = uuid.uuid4().hex
    heatmap_file = f"{file_id}.png"
    mask_file = f"{file_id}.png"

    cv2.imwrite(os.path.join(heatmaps_dir, heatmap_file), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(masks_dir, mask_file), mask)

    return {
        "stageA_label": stageA_label,
        "stageA_confidence": confA,
        "label": final_label,
        "confidence": confB,
        "heatmap_file": heatmap_file,
        "mask_file": mask_file,
        "bbox": bbox,
        "best_layer": best_layer
    }