import io
import os
import uuid
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image, ImageChops
from torchvision import transforms

IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "authentic",
    "copy_move",
    "enhancement",
    "removal_inpainting",
    "splicing",
]

ELA_QUALITIES = (70, 85, 95)
MIN_COMPONENT_AREA = 80
OVERLAY_ALPHA = 0.55

NORMALIZE_9 = transforms.Normalize(
    mean=[0.485, 0.456, 0.406, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    std=[0.229, 0.224, 0.225, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
)


def _ela_single(img_rgb: Image.Image, quality: int) -> np.ndarray:
    buf = io.BytesIO()
    img_rgb.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    recomp = Image.open(buf).convert("RGB")
    diff = ImageChops.difference(img_rgb, recomp)
    arr = np.array(diff, dtype=np.float32)
    mx = arr.max()
    if mx > 0:
        arr = arr * (255.0 / mx)
    return arr


def compute_ela_multiscale(img: Image.Image) -> Image.Image:
    img_rgb = img.convert("RGB")
    maps = [_ela_single(img_rgb, q) for q in ELA_QUALITIES]
    avg = np.clip(np.mean(maps, axis=0), 0, 255).astype(np.uint8)
    return Image.fromarray(avg)


def compute_noise_map_rgb(img: Image.Image) -> Image.Image:
    rgb = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    blurred = cv2.GaussianBlur(rgb, (5, 5), sigmaX=1.5)
    noise = rgb - blurred

    out = np.zeros_like(noise)
    for c in range(3):
        p2, p98 = np.percentile(noise[..., c], [2, 98])
        denom = max(p98 - p2, 1e-6)
        out[..., c] = np.clip((noise[..., c] - p2) / denom, 0.0, 1.0)

    return Image.fromarray((out * 255).astype(np.uint8), mode="RGB")


def preprocess_9ch(image_path: str):
    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    ela = compute_ela_multiscale(img).resize(IMG_SIZE)
    noise = compute_noise_map_rgb(img).resize(IMG_SIZE)

    rgb_np = np.asarray(img, dtype=np.float32) / 255.0
    ela_np = np.asarray(ela, dtype=np.float32) / 255.0
    noise_np = np.asarray(noise, dtype=np.float32) / 255.0

    rgb_t = torch.from_numpy(rgb_np).permute(2, 0, 1)      # 3,H,W
    ela_t = torch.from_numpy(ela_np).permute(2, 0, 1)      # 3,H,W
    noise_t = torch.from_numpy(noise_np).permute(2, 0, 1)  # 3,H,W

    x = torch.cat([rgb_t, ela_t, noise_t], dim=0)          # 9,H,W
    x = NORMALIZE_9(x).unsqueeze(0).float()                # 1,9,H,W

    return img, ela, noise, x


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.h1 = target_layer.register_forward_hook(self._save_activation)
        self.h2 = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def remove(self):
        self.h1.remove()
        self.h2.remove()

    def generate(self, x, class_idx):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        score = logits[:, class_idx]
        score.backward(retain_graph=True)

        grads = self.gradients
        acts = self.activations
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=IMG_SIZE, mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


@torch.no_grad()
def predict_logits(model, x, device):
    x = x.to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    return logits, probs


def cam_to_mask(cam01, keep_percent=10):
    cam_r = np.clip(cam01, 0, 1)
    thr = np.percentile(cam_r, 100 - keep_percent)
    mask = (cam_r >= thr).astype(np.uint8) * 255

    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask


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


def overlay_heatmap(original_rgb, cam01, mask255, alpha=0.55):
    orig = np.array(original_rgb)
    heat_u8 = np.uint8(cam01 * 255)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

    mask3 = cv2.cvtColor(mask255, cv2.COLOR_GRAY2BGR)
    heat_color = cv2.bitwise_and(heat_color, mask3)

    orig_bgr = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)
    out = cv2.addWeighted(orig_bgr, 1 - alpha, heat_color, alpha, 0)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def draw_bbox(rgb, contour):
    if contour is None:
        return rgb, None
    x, y, w, h = cv2.boundingRect(contour)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.rectangle(bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}


def save_rgb(path, pil_img):
    np_img = np.array(pil_img)
    cv2.imwrite(path, cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR))


def predict_image(image_path: str, model, device, outputs_dir: str):
    original, ela_img, noise_img, x = preprocess_9ch(image_path)

    originals_dir = os.path.join(outputs_dir, "originals")
    ela_dir = os.path.join(outputs_dir, "ela")
    noise_dir = os.path.join(outputs_dir, "noise")
    heatmaps_dir = os.path.join(outputs_dir, "heatmaps")
    masks_dir = os.path.join(outputs_dir, "masks")

    os.makedirs(originals_dir, exist_ok=True)
    os.makedirs(ela_dir, exist_ok=True)
    os.makedirs(noise_dir, exist_ok=True)
    os.makedirs(heatmaps_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    file_id = uuid.uuid4().hex

    original_file = f"{file_id}.png"
    ela_file = f"{file_id}.png"
    noise_file = f"{file_id}.png"

    save_rgb(os.path.join(originals_dir, original_file), original)
    save_rgb(os.path.join(ela_dir, ela_file), ela_img)
    save_rgb(os.path.join(noise_dir, noise_file), noise_img)

    logits, probs = predict_logits(model, x, device)
    probs_np = probs.squeeze(0).cpu().numpy()

    idx = int(np.argmax(probs_np))
    conf = float(probs_np[idx]) * 100.0
    pred_label = CLASS_NAMES[idx]

    stageA_label = "authentic" if pred_label == "authentic" else "tampered"
    stageA_conf = conf if pred_label == "authentic" else float(100.0 - probs_np[0] * 100.0)

    if pred_label == "authentic":
        return {
            "stageA_label": stageA_label,
            "stageA_confidence": stageA_conf,
            "label": "N/A",
            "confidence": 0.0,
            "original_file": original_file,
            "ela_file": ela_file,
            "noise_file": noise_file,
            "heatmap_file": None,
            "mask_file": None,
            "bbox": None,
            "best_layer": None,
        }

    target_layer = model.layer4[0][-1].conv3
    cam_engine = GradCAM(model, target_layer)

    x = x.to(device)
    model.zero_grad(set_to_none=True)
    logits = model(x)
    class_score = logits[:, idx]
    class_score.backward()

    cam01 = cam_engine.generate(x, idx)
    cam_engine.remove()

    raw_mask = cam_to_mask(cam01, keep_percent=10)
    mask, contour = keep_largest_component(raw_mask, min_area=MIN_COMPONENT_AREA)

    overlay = overlay_heatmap(original, cam01, mask, alpha=OVERLAY_ALPHA)
    overlay, bbox = draw_bbox(overlay, contour)

    heatmap_file = f"{file_id}.png"
    mask_file = f"{file_id}.png"

    cv2.imwrite(
        os.path.join(heatmaps_dir, heatmap_file),
        cv2.cvtColor(np.array(overlay), cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(os.path.join(masks_dir, mask_file), mask)

    return {
        "stageA_label": stageA_label,
        "stageA_confidence": stageA_conf,
        "label": pred_label,
        "confidence": conf,
        "original_file": original_file,
        "ela_file": ela_file,
        "noise_file": noise_file,
        "heatmap_file": heatmap_file,
        "mask_file": mask_file,
        "bbox": bbox,
        "best_layer": "layer4_last_conv3",
    }