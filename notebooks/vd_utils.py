"""Shared helpers for the VisDrone-VID detection

Conventions
* Class ids are 0-based (0=person ... 7=motor), as in YOLO labels and the submission.
  COCO json uses 1-based category ids; torchvision models use 0 for background,
  so they see cls + 1.
* Boxes are in ORIGINAL image pixels unless a function says otherwise.
* A frame is identified by ``key = "<clip>/<frame stem>"``.
"""
from __future__ import annotations

import contextlib
import glob
import io
import json
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

CLASSES = ["person", "bicycle", "car", "van", "truck", "tricycle", "bus", "motor"]
NUM_CLASSES = len(CLASSES)
COMP_SLUG = "multiple-object-video-detection-hw-2-ucu-ai-2026"
GRAY = 114          # fill value for masked ignore regions (same gray Ultralytics uses for padding)
SEED = 42
MAX_DET = 500       # the leaderboard scores at most 500 detections per image


def _search(name: str, root: str = "/kaggle/input", max_depth: int = 5):
    """Breadth-first glob for `name` under `root` (avoids a slow recursive walk)."""
    for depth in range(max_depth + 1):
        hits = sorted(glob.glob(os.path.join(root, *["*"] * depth, name)))
        if hits:
            return hits[0]
    return None


def comp_root() -> str:
    """Root folder of the competition data (contains train/, val/, test/)."""
    if os.environ.get("VD_COMP_ROOT"):
        return os.environ["VD_COMP_ROOT"]
    hit = _search(COMP_SLUG, max_depth=2)
    if hit is None:
        raise FileNotFoundError("Competition data not found: add it as notebook input.")
    return hit


def prep_root() -> str:
    """Output folder of notebook 00 (frames.csv, annotations, exported frames)."""
    if os.environ.get("VD_PREP_ROOT"):
        return os.environ["VD_PREP_ROOT"]
    if os.path.exists("/kaggle/working/prep/frames.csv"):
        return "/kaggle/working/prep"
    # walk every input except the (huge) competition folder; prep = folder with frames.csv + annotations
    for top in sorted(os.listdir("/kaggle/input")) if os.path.isdir("/kaggle/input") else []:
        if top == "competitions" or COMP_SLUG in top:
            continue
        for dirpath, dirnames, filenames in os.walk(os.path.join("/kaggle/input", top)):
            dirnames[:] = [d for d in dirnames if d not in ("train_masked", "figures", "runs")]
            if "frames.csv" in filenames and "annotations.csv.gz" in filenames:
                return dirpath
    inputs = sorted(glob.glob("/kaggle/input/*/*/*"))
    raise FileNotFoundError("Preprocessed data (frames.csv) not found. Attach the output of a COMMITTED "
                            f"version of notebook 00 as input. Attached inputs: {inputs}")


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _resolve_rel_paths(imgs: pd.DataFrame, root: str, split: str) -> list[str]:
    """COCO file_name may be relative to the split or to images/; find the rule that works."""
    rules = [
        lambda r: os.path.join(split, r.file_name),
        lambda r: os.path.join(split, "images", r.file_name),
        lambda r: os.path.join(split, "images", r.clip, os.path.basename(r.file_name)),
        lambda r: r.file_name,
    ]
    first = next(imgs.itertuples())
    for rule in rules:
        if os.path.exists(os.path.join(root, rule(first))):
            return [os.path.normpath(rule(r)) for r in imgs.itertuples()]
    raise FileNotFoundError(f"Cannot resolve image path for {first.file_name!r}")


def load_coco_split(split: str, root: str | None = None):
    """Read <split>/annotations_coco.json into two tables.

    frames: key, split, clip, stem, frame_idx (0-based order in the clip), orig_w, orig_h, img_rel
    anns:   key, cls (-1 for ignore regions), x, y, w, h, iscrowd, occlusion, truncation, track_id
    """
    root = root or comp_root()
    with open(os.path.join(root, split, "annotations_coco.json")) as f:
        coco = json.load(f)

    imgs = pd.DataFrame(coco["images"]).rename(columns={"id": "coco_img_id"})
    if "video" not in imgs:
        imgs["video"] = imgs["file_name"].map(lambda p: Path(p).parent.name)
    imgs["clip"] = imgs["video"].astype(str)
    imgs["stem"] = imgs["file_name"].map(lambda p: Path(p).stem)
    imgs["split"] = split
    imgs["img_rel"] = _resolve_rel_paths(imgs, root, split)
    imgs["key"] = imgs["clip"] + "/" + imgs["stem"]
    order = "frame_id" if "frame_id" in imgs else "stem"
    imgs = imgs.sort_values(["clip", order]).reset_index(drop=True)
    imgs["frame_idx"] = imgs.groupby("clip").cumcount()
    frames = imgs.rename(columns={"width": "orig_w", "height": "orig_h"})[
        ["key", "split", "clip", "stem", "frame_idx", "orig_w", "orig_h", "img_rel", "coco_img_id"]]

    anns = pd.DataFrame(coco["annotations"])
    bb = np.asarray(anns["bbox"].tolist(), dtype=np.float64).reshape(-1, 4)
    anns["x"], anns["y"], anns["w"], anns["h"] = bb.T
    anns["iscrowd"] = anns["iscrowd"].fillna(0).astype(int) if "iscrowd" in anns else 0
    anns["cls"] = np.where(anns["iscrowd"] == 1, -1, anns["category_id"].astype(int) - 1)
    for col in ["occlusion", "truncation", "track_id"]:
        if col not in anns:
            anns[col] = -1
    anns["key"] = anns["image_id"].map(dict(zip(frames.coco_img_id, frames.key)))
    anns = anns[["key", "cls", "x", "y", "w", "h", "iscrowd", "occlusion", "truncation", "track_id"]]
    return frames.drop(columns="coco_img_id"), anns.reset_index(drop=True)


def load_prep(prep: str | None = None):
    """frames.csv and annotations.csv.gz written by notebook 00."""
    prep = prep or prep_root()
    text = {"key": str, "clip": str, "stem": str, "img_rel": str}   # keep "0000001" as text
    frames = pd.read_csv(os.path.join(prep, "frames.csv"), dtype=text)
    anns = pd.read_csv(os.path.join(prep, "annotations.csv.gz"), dtype={"key": str})
    return frames, anns


def select_frames(frames: pd.DataFrame, folds=None, step: int = 1) -> pd.DataFrame:
    """Frames of the given folds, keeping every `step`-th frame of each clip."""
    sel = frames[frames.frame_idx % step == 0]
    if folds is not None:
        sel = sel[sel.fold.isin(list(np.atleast_1d(folds)))]
    return sel.reset_index(drop=True)


def train_image_paths(frames: pd.DataFrame, prep: str | None = None) -> list[str]:
    """Exported (masked + resized) training images."""
    prep = prep or prep_root()
    return [os.path.join(prep, "train_masked", "images", c, s + ".jpg")
            for c, s in zip(frames["clip"], frames["stem"])]


def orig_image_paths(frames: pd.DataFrame, comp: str | None = None) -> list[str]:
    """Original competition images (used for validation and test)."""
    comp = comp or comp_root()
    return [os.path.join(comp, p) for p in frames.img_rel]


def label_path(img_path: str) -> str:
    """YOLO convention: .../images/<clip>/<f>.jpg -> .../labels/<clip>/<f>.txt"""
    head, tail = img_path.rsplit(os.sep + "images" + os.sep, 1)
    return os.path.join(head, "labels", os.path.splitext(tail)[0] + ".txt")


def read_yolo_labels(path: str, img_w: int, img_h: int):
    """Return (boxes_xyxy in pixels of this image, cls 0-based)."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.int64)
    a = np.loadtxt(path, ndmin=2)
    cls, cx, cy, w, h = a[:, 0].astype(np.int64), a[:, 1] * img_w, a[:, 2] * img_h, a[:, 3] * img_w, a[:, 4] * img_h
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1).astype(np.float32)
    return boxes, cls


def box_coverage(boxes: np.ndarray, regions: np.ndarray) -> np.ndarray:
    """Fraction of each (x, y, w, h) box lying inside the ignore regions (capped at 1)."""
    if len(boxes) == 0 or len(regions) == 0:
        return np.zeros(len(boxes))
    b, r = boxes[:, None, :], regions[None, :, :]
    ix = np.clip(np.minimum(b[..., 0] + b[..., 2], r[..., 0] + r[..., 2]) - np.maximum(b[..., 0], r[..., 0]), 0, None)
    iy = np.clip(np.minimum(b[..., 1] + b[..., 3], r[..., 1] + r[..., 3]) - np.maximum(b[..., 1], r[..., 1]), 0, None)
    return np.clip((ix * iy).sum(1) / np.maximum(boxes[:, 2] * boxes[:, 3], 1e-9), 0, 1)


def export_frame(task: dict) -> dict:
    """Mask ignore regions, resize to `long_side`, write JPEG + YOLO label file.

    task keys: key, src, dst_img, dst_lbl, boxes (n,5: cls x y w h), regions (m,4),
               long_side, cover_thr
    Boxes mostly hidden by a mask (coverage >= cover_thr) or smaller than 2 px after
    resizing are dropped, so the model is never asked to find an object it cannot see.
    """
    img = cv2.imread(task["src"])
    if img is None:
        raise IOError(f"Cannot read {task['src']}")
    h0, w0 = img.shape[:2]
    regions = np.asarray(task["regions"], dtype=np.float64).reshape(-1, 4)
    for x, y, w, h in regions:
        x0, y0 = max(int(np.floor(x)), 0), max(int(np.floor(y)), 0)
        x1, y1 = min(int(np.ceil(x + w)), w0), min(int(np.ceil(y + h)), h0)
        img[y0:y1, x0:x1] = GRAY

    s = min(1.0, task["long_side"] / max(w0, h0))
    if s < 1.0:
        img = cv2.resize(img, (round(w0 * s), round(h0 * s)), interpolation=cv2.INTER_AREA)
    h1, w1 = img.shape[:2]
    os.makedirs(os.path.dirname(task["dst_img"]), exist_ok=True)
    cv2.imwrite(task["dst_img"], img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    boxes = np.asarray(task["boxes"], dtype=np.float64).reshape(-1, 5)
    x0 = np.clip(boxes[:, 1], 0, w0)
    y0 = np.clip(boxes[:, 2], 0, h0)
    x1 = np.clip(boxes[:, 1] + boxes[:, 3], 0, w0)
    y1 = np.clip(boxes[:, 2] + boxes[:, 4], 0, h0)
    bw, bh = x1 - x0, y1 - y0
    cov = box_coverage(np.stack([x0, y0, bw, bh], 1), regions)
    keep = (bw * s >= 2) & (bh * s >= 2) & (cov < task["cover_thr"])
    lines = [f"{int(c)} {(a + w / 2) / w0:.6f} {(b + h / 2) / h0:.6f} {w / w0:.6f} {h / h0:.6f}"
             for c, a, b, w, h in zip(boxes[keep, 0], x0[keep], y0[keep], bw[keep], bh[keep])]
    os.makedirs(os.path.dirname(task["dst_lbl"]), exist_ok=True)
    with open(task["dst_lbl"], "w") as f:
        f.write("\n".join(lines))
    return {"key": task["key"], "new_w": w1, "new_h": h1,
            "n_kept": int(keep.sum()), "n_dropped": int((~keep).sum())}


def write_yolo_data(train_imgs: list[str], val_imgs: list[str], out_dir: str = "/kaggle/working/yolo_data") -> str:
    """Write image lists + data.yaml for Ultralytics; labels are found via /images/ -> /labels/."""
    os.makedirs(out_dir, exist_ok=True)
    for name, paths in [("train", train_imgs), ("val", val_imgs)]:
        with open(os.path.join(out_dir, f"{name}.txt"), "w") as f:
            f.write("\n".join(paths))
    yaml_path = os.path.join(out_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {out_dir}\ntrain: train.txt\nval: val.txt\nnames:\n")
        f.writelines(f"  {i}: {n}\n" for i, n in enumerate(CLASSES))
    return yaml_path


def make_coco_gt(frames: pd.DataFrame, anns: pd.DataFrame, out_path: str):
    """COCO ground truth for the given frames in original pixels.

    Ignore regions are written with iscrowd=1 once per category: COCOeval matches crowd
    regions per class, so a class-less region must exist for every class to be ignored
    for all of them. Returns (json path, {key: image_id}).
    """
    frames = frames.reset_index(drop=True)
    key2id = {k: i + 1 for i, k in enumerate(frames.key)}
    images = [{"id": key2id[r.key], "file_name": r.img_rel, "width": int(r.orig_w),
               "height": int(r.orig_h)} for r in frames.itertuples()]
    a = anns[anns.key.isin(key2id)]
    annotations = []
    for r in a[a.iscrowd == 0].itertuples():
        annotations.append({"image_id": key2id[r.key], "category_id": int(r.cls) + 1,
                            "bbox": [r.x, r.y, r.w, r.h], "area": r.w * r.h, "iscrowd": 0})
    for r in a[a.iscrowd == 1].itertuples():
        for c in range(1, NUM_CLASSES + 1):
            annotations.append({"image_id": key2id[r.key], "category_id": c,
                                "bbox": [r.x, r.y, r.w, r.h], "area": r.w * r.h, "iscrowd": 1})
    for i, ann in enumerate(annotations):
        ann["id"] = i + 1
    coco = {"info": {}, "licenses": [], "images": images, "annotations": annotations,
            "categories": [{"id": i + 1, "name": n} for i, n in enumerate(CLASSES)]}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(coco, f)
    return out_path, key2id


def to_coco_dets(image_id: int, boxes_xyxy, scores, labels0, max_det: int = MAX_DET) -> list[dict]:
    """Convert one image's predictions (original pixels, 0-based classes) to COCO results."""
    boxes_xyxy, scores, labels0 = np.asarray(boxes_xyxy), np.asarray(scores), np.asarray(labels0)
    valid = (labels0 >= 0) & (labels0 < NUM_CLASSES)
    boxes_xyxy, scores, labels0 = boxes_xyxy[valid], scores[valid], labels0[valid]
    order = np.argsort(-scores)[:max_det]
    return [{"image_id": int(image_id), "category_id": int(labels0[i]) + 1,
             "bbox": [float(boxes_xyxy[i, 0]), float(boxes_xyxy[i, 1]),
                      float(boxes_xyxy[i, 2] - boxes_xyxy[i, 0]), float(boxes_xyxy[i, 3] - boxes_xyxy[i, 1])],
             "score": float(scores[i])} for i in order]


def coco_evaluate(gt_json: str, dets: list[dict]) -> dict:
    """COCO bbox metrics with up to 500 detections per image, plus per-class AP."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    keys = ["mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR10", "AR100", "AR500"]
    if not dets:
        return {k: 0.0 for k in keys}
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(gt_json)
        ev = COCOeval(gt, gt.loadRes(dets), "bbox")
        ev.params.maxDets = [10, 100, MAX_DET]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    s = ev.stats
    res = dict(zip(keys, [s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8]]))
    prec = ev.eval["precision"]                      # [T, R, K, A, M]
    for k, cat_id in enumerate(ev.params.catIds):
        p = prec[:, :, k, 0, -1]
        res[f"AP_{CLASSES[cat_id - 1]}"] = float(p[p > -1].mean()) if (p > -1).any() else float("nan")
    return {k: round(float(v), 4) for k, v in res.items()}


def measure_fps(predict_fn, inputs: list, warmup: int = 10, n: int = 100) -> float:
    """Images per second for `predict_fn(single_input)` (batch size 1)."""
    try:
        import torch
        sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    except ImportError:
        sync = lambda: None
    for x in inputs[:warmup]:
        predict_fn(x)
    runs = inputs[warmup:warmup + n]
    sync()
    t0 = time.perf_counter()
    for x in runs:
        predict_fn(x)
    sync()
    return len(runs) / (time.perf_counter() - t0)


def prediction_string(boxes_xyxy, scores, labels0, max_det: int = MAX_DET) -> str:
    """'class score x y w h ...' with (x, y) the top-left corner in original pixels."""
    boxes_xyxy, scores, labels0 = np.asarray(boxes_xyxy), np.asarray(scores), np.asarray(labels0)
    valid = (labels0 >= 0) & (labels0 < NUM_CLASSES)
    boxes_xyxy, scores, labels0 = boxes_xyxy[valid], scores[valid], labels0[valid]
    order = np.argsort(-scores)[:max_det]
    return " ".join(
        f"{int(labels0[i])} {scores[i]:.4f} {boxes_xyxy[i, 0]:.1f} {boxes_xyxy[i, 1]:.1f} "
        f"{boxes_xyxy[i, 2] - boxes_xyxy[i, 0]:.1f} {boxes_xyxy[i, 3] - boxes_xyxy[i, 1]:.1f}"
        for i in order)


def write_submission(pred: dict, out_path: str = "/kaggle/working/submission.csv") -> pd.DataFrame:
    """`pred` maps image_id -> PredictionString; missing images get an empty string.

    Row order follows sample_submission*.csv if it exists anywhere in the competition data,
    otherwise test_images.csv (same image ids, one row per test frame).
    """
    root = comp_root()
    samples = sorted(glob.glob(os.path.join(root, "sample_submission*.csv")) +
                     glob.glob(os.path.join(root, "*", "sample_submission*.csv")))
    ids = pd.read_csv(samples[0] if samples else os.path.join(root, "test_images.csv")).image_id
    sub = pd.DataFrame({"image_id": ids, "PredictionString": ids.map(pred).fillna("")})
    sub.to_csv(out_path, index=False)
    return sub


def wh_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between (w, h) pairs aligned at a common centre: [len(a), len(b)]."""
    inter = np.minimum(a[:, None, 0], b[None, :, 0]) * np.minimum(a[:, None, 1], b[None, :, 1])
    return inter / (a[:, None].prod(2) + b[None].prod(2) - inter)


def kmeans_anchors(wh: np.ndarray, k: int, iters: int = 100, seed: int = SEED):
    """YOLOv2-style k-means with 1 - IoU distance. Returns (centres sorted by area, mean best IoU)."""
    rng = np.random.default_rng(seed)
    centres = wh[rng.choice(len(wh), k, replace=False)]
    for _ in range(iters):
        assign = wh_iou(wh, centres).argmax(1)
        new = np.array([np.median(wh[assign == j], 0) if (assign == j).any() else centres[j] for j in range(k)])
        if np.allclose(new, centres):
            break
        centres = new
    centres = centres[np.argsort(centres.prod(1))]
    return centres, float(wh_iou(wh, centres).max(1).mean())


def anchor_grid_iou(boxes_xywh: np.ndarray, strides, sizes_per_level, ratios) -> np.ndarray:
    """Best IoU of each GT box with any anchor placed on the real grid (centres at multiples of the stride).

    Unlike k-means IoU, which assumes anchor and box share a centre, this accounts for the offset between
    the box centre and the nearest anchor centre, so anchors much smaller than their stride score low.
    """
    b = np.asarray(boxes_xywh, np.float64).reshape(-1, 4)
    cx, cy, w, h = b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2, b[:, 2], b[:, 3]
    best = np.zeros(len(b))
    for stride, sizes in zip(strides, sizes_per_level):
        dx = cx - np.round(cx / stride) * stride
        dy = cy - np.round(cy / stride) * stride
        for aw, ah in anchor_wh(np.atleast_1d(sizes), ratios):
            ix = np.clip(np.minimum(dx + w / 2, aw / 2) - np.maximum(dx - w / 2, -aw / 2), 0, None)
            iy = np.clip(np.minimum(dy + h / 2, ah / 2) - np.maximum(dy - h / 2, -ah / 2), 0, None)
            inter = ix * iy
            best = np.maximum(best, inter / (w * h + aw * ah - inter))
    return best


def anchor_wh(sizes, ratios) -> np.ndarray:
    """(w, h) of torchvision-style anchors; ratio = h / w."""
    return np.array([[s / np.sqrt(r), s * np.sqrt(r)] for s in sizes for r in ratios])


def draw_boxes(img_bgr: np.ndarray, boxes_xyxy, labels0=None, regions=None, thickness: int = 2) -> np.ndarray:
    """Return an RGB copy with boxes (coloured by class) and ignore regions (hatched gray)."""
    img = img_bgr.copy()
    if regions is not None and len(regions):
        overlay = img.copy()
        for x, y, w, h in np.asarray(regions):
            cv2.rectangle(overlay, (int(x), int(y)), (int(x + w), int(y + h)), (200, 200, 200), -1)
        img = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)
    palette = [(56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
               (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61)]
    for i, (x0, y0, x1, y1) in enumerate(np.asarray(boxes_xyxy).reshape(-1, 4)):
        c = palette[int(labels0[i]) % 8] if labels0 is not None else (0, 255, 0)
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), c, thickness)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def draw_errors(img_bgr: np.ndarray, err: dict, regions=None, thickness: int = 2) -> np.ndarray:
    """Matched GT green, missed GT yellow, false positives red (from `frame_errors`)."""
    img = draw_boxes(img_bgr, np.zeros((0, 4)), regions=regions)[..., ::-1].copy()
    for boxes, colour in [(err["gt"][err["tp_gt"]], (0, 200, 0)), (err["gt"][~err["tp_gt"]], (0, 230, 255)),
                          (err["fp_boxes"], (0, 0, 255))]:
        for x0, y0, x1, y1 in np.asarray(boxes).reshape(-1, 4):
            cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), colour, thickness)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def box_iou_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between xyxy boxes: [len(a), len(b)]."""
    a, b = np.asarray(a, np.float64).reshape(-1, 4), np.asarray(b, np.float64).reshape(-1, 4)
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / np.maximum(area(a)[:, None] + area(b)[None, :] - inter, 1e-9)


def frame_errors(gt_xyxy, gt_cls, regions_xywh, det_xyxy, det_scores, det_cls,
                 score_thr: float = 0.3, iou_thr: float = 0.5) -> dict:
    """Simplified TIDE-style breakdown for one frame at a fixed score threshold.

    Detections are matched greedily (highest score first) to unmatched GT of the same class.
    Unmatched detections are typed:
      ignore - mostly inside an ignore region (not an error)
      dup    - IoU >= thr with an already matched GT of the same class
      cls    - IoU >= thr with a GT of another class
      loc    - 0.1 <= IoU < thr with a GT of the same class
      bkg    - everything else
    Unmatched GT are missed objects (FN).
    """
    gt, gc = np.asarray(gt_xyxy, np.float64).reshape(-1, 4), np.asarray(gt_cls).reshape(-1)
    keep = np.asarray(det_scores) >= score_thr
    d, dc, ds = np.asarray(det_xyxy, np.float64).reshape(-1, 4)[keep], np.asarray(det_cls)[keep], np.asarray(det_scores)[keep]
    order = np.argsort(-ds)
    d, dc = d[order], dc[order]
    iou = box_iou_np(d, gt)
    tp_gt = np.zeros(len(gt), bool)
    fp_type, fp_idx = [], []
    cover = box_coverage(np.c_[d[:, :2], d[:, 2:] - d[:, :2]], np.asarray(regions_xywh).reshape(-1, 4))
    for i in range(len(d)):
        same = gc == dc[i]
        cand = np.where(same & ~tp_gt & (iou[i] >= iou_thr))[0] if len(gt) else []
        if len(cand):
            tp_gt[cand[np.argmax(iou[i, cand])]] = True
            continue
        if cover[i] > 0.5:
            t = "ignore"
        elif len(gt) and (iou[i] * same * tp_gt).max(initial=0) >= iou_thr:
            t = "dup"
        elif len(gt) and (iou[i] * ~same).max(initial=0) >= iou_thr:
            t = "cls"
        elif len(gt) and (iou[i] * same).max(initial=0) >= 0.1:
            t = "loc"
        else:
            t = "bkg"
        fp_type.append(t)
        fp_idx.append(i)
    fp_type = np.array(fp_type, dtype=object)
    real = fp_type != "ignore"
    return {"gt": gt, "gt_cls": gc, "tp_gt": tp_gt, "fp_boxes": d[fp_idx][real] if fp_idx else np.zeros((0, 4)),
            "fp_type": fp_type[real], "n_tp": int(tp_gt.sum()), "n_fn": int((~tp_gt).sum())}


def find_run_file(name: str, filename: str, work: str = "/kaggle/working") -> str | None:
    """`<runs dir>/<name>/<filename>` from this session or from an earlier version attached as input."""
    for local in glob.glob(os.path.join(work, "*", name, filename)):
        return local
    for top in sorted(os.listdir("/kaggle/input")) if os.path.isdir("/kaggle/input") else []:
        if top == "competitions" or COMP_SLUG in top:
            continue
        for dirpath, dirnames, filenames in os.walk(os.path.join("/kaggle/input", top)):
            dirnames[:] = [x for x in dirnames if x not in ("train_masked", "images", "labels")]
            if os.path.basename(dirpath) == name and filename in filenames:
                return os.path.join(dirpath, filename)
    return None


def load_result(name: str, work: str = "/kaggle/working") -> dict | None:
    """Holdout metrics `results/<name>.json` saved by any detector notebook (this session or an attached output)."""
    local = os.path.join(work, "results", f"{name}.json")
    path = local if os.path.exists(local) else find_run_file("results", f"{name}.json", work=work)
    return json.load(open(path)) if path else None


class DetDataset:
    """Map-style dataset: image tensor in [0, 1] + target with xyxy pixel boxes and labels cls + 1.

    Normalisation (ImageNet mean/std) and resizing are done inside the torchvision model
    (GeneralizedRCNNTransform). Augmentation: horizontal flip only.
    """

    def __init__(self, img_paths: list[str], train: bool = False, hflip: float = 0.5):
        self.paths, self.train, self.hflip = list(img_paths), train, hflip

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        import torch
        img = cv2.cvtColor(cv2.imread(self.paths[i]), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        boxes, cls = read_yolo_labels(label_path(self.paths[i]), w, h)
        if self.train and random.random() < self.hflip:
            img = img[:, ::-1]
            boxes = np.stack([w - boxes[:, 2], boxes[:, 1], w - boxes[:, 0], boxes[:, 3]], 1) if len(boxes) else boxes
        tensor = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div_(255)
        target = {"boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                  "labels": torch.as_tensor(cls + 1, dtype=torch.int64)}
        return tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


def _worker_init(worker_id: int):
    import torch
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)
    cv2.setNumThreads(0)


def build_detector(kind: str, **kw):
    """COCO-pretrained torchvision detector with a fresh 9-class head.

    kind = "faster_rcnn": kw may set anchor_sizes (one size per FPN level, 5 levels),
           rpn_{pre,post}_nms_top_n_{train,test}, box_batch_size_per_image, box_positive_fraction.
    kind = "retinanet":   kw may set anchor_sizes (tuple of tuples, 5 levels), aspect_ratios,
           focal_alpha, focal_gamma, bbox_reg_loss ("l1", "smooth_l1", "giou", "diou", "ciou"),
           fg_iou_thresh, bg_iou_thresh.
    Common: min_size / max_size (input resize), detections per image = MAX_DET.
    """
    import torchvision.models.detection as D
    from torchvision.models.detection.anchor_utils import AnchorGenerator

    kw = dict(kw)
    anchor_sizes = kw.pop("anchor_sizes", None)
    aspect_ratios = kw.pop("aspect_ratios", (0.5, 1.0, 2.0))
    common = dict(min_size=kw.pop("min_size", 1344), max_size=kw.pop("max_size", 1344))

    if kind == "faster_rcnn":
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        model = D.fasterrcnn_resnet50_fpn(weights=D.FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
                                          box_detections_per_img=MAX_DET, box_score_thresh=0.001, **common, **kw)
        if anchor_sizes is not None:
            model.rpn.anchor_generator = AnchorGenerator(tuple((s,) for s in anchor_sizes),
                                                         (tuple(aspect_ratios),) * len(anchor_sizes))
        in_f = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_f, NUM_CLASSES + 1)
        return model

    if kind == "retinanet":
        from functools import partial
        import torch
        from torchvision.models.detection.retinanet import RetinaNetClassificationHead, RetinaNetRegressionHead
        focal = dict(alpha=kw.pop("focal_alpha", 0.25), gamma=kw.pop("focal_gamma", 2.0))
        reg_loss = kw.pop("bbox_reg_loss", "giou")      # torchvision's v2 recipe uses GIoU
        model = D.retinanet_resnet50_fpn_v2(weights=D.RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1,
                                            detections_per_img=MAX_DET, score_thresh=0.001, **common, **kw)
        if anchor_sizes is not None:
            model.anchor_generator = AnchorGenerator(tuple(tuple(s) for s in anchor_sizes),
                                                     (tuple(aspect_ratios),) * len(anchor_sizes))
        norm = partial(torch.nn.GroupNorm, 32)
        n_anchors = model.anchor_generator.num_anchors_per_location()[0]
        if n_anchors != model.head.regression_head.bbox_reg.out_channels // 4:
            model.head.regression_head = RetinaNetRegressionHead(256, n_anchors, norm_layer=norm)
        cls_head = RetinaNetClassificationHead(256, n_anchors, NUM_CLASSES + 1, norm_layer=norm)
        model.head.classification_head = cls_head
        _set_focal(cls_head, **focal)
        model.head.regression_head._loss_type = reg_loss  # set explicitly: a replaced head defaults to L1
        return model

    raise ValueError(kind)


def _set_focal(cls_head, alpha: float, gamma: float):
    """Replace the focal-loss parameters of a torchvision RetinaNet classification head."""
    import types
    from torchvision.ops import sigmoid_focal_loss
    from torchvision.models.detection.retinanet import _sum

    def compute_loss(self, targets, head_outputs, matched_idxs):
        import torch
        losses = []
        cls_logits = head_outputs["cls_logits"]
        for tgt, logits, idxs in zip(targets, cls_logits, matched_idxs):
            fg = idxs >= 0
            n_fg = fg.sum()
            gt = torch.zeros_like(logits)
            gt[fg, tgt["labels"][idxs[fg]]] = 1.0
            valid = idxs != self.BETWEEN_THRESHOLDS
            losses.append(sigmoid_focal_loss(logits[valid], gt[valid], alpha=alpha, gamma=gamma,
                                             reduction="sum") / max(1, n_fg))
        return _sum(losses) / len(targets)

    cls_head.compute_loss = types.MethodType(compute_loss, cls_head)


def _train_entry(cfg: dict):
    """Process entry point: train one config on one GPU, write log/weights/info into cfg['out_dir']."""
    import traceback
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # less fragmentation at 1344 px
    os.makedirs(cfg["out_dir"], exist_ok=True)
    try:
        train_torchvision(cfg)
    except Exception:
        with open(os.path.join(cfg["out_dir"], "error.txt"), "w") as f:
            f.write(traceback.format_exc())
        raise


def train_torchvision(cfg: dict) -> dict:
    """SGD + AMP fine-tuning of a torchvision detector.

    cfg keys: name, kind, model_kwargs, train_paths, epochs, batch, lr, workers, seed, device, out_dir.
    Schedule: linear warmup (500 iterations) then cosine decay to 0; weight decay 1e-4; grad clip 10.
    """
    import math
    import warnings
    import torch
    # the first optimizer step is skipped by GradScaler while it calibrates; the scheduler warning is harmless
    warnings.filterwarnings("ignore", message=r"Detected call of `lr_scheduler\.step\(\)`")
    cv2.setNumThreads(0)
    seed_everything(cfg["seed"])
    device = torch.device(cfg["device"])
    torch.cuda.set_device(device)
    model = build_detector(cfg["kind"], **cfg["model_kwargs"]).to(device)
    loader = torch.utils.data.DataLoader(
        DetDataset(cfg["train_paths"], train=True), batch_size=cfg["batch"], shuffle=True,
        num_workers=cfg["workers"], collate_fn=collate_fn, drop_last=True, pin_memory=True,
        worker_init_fn=_worker_init, generator=torch.Generator().manual_seed(cfg["seed"]))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=cfg["lr"], momentum=0.9, weight_decay=1e-4)
    total, warm = cfg["epochs"] * len(loader), min(500, len(loader))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: (it + 1) / warm if it < warm else 0.5 * (1 + math.cos(math.pi * (it - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda")
    log_path = os.path.join(cfg["out_dir"], "log.json")
    log, t0 = [], time.time()
    for epoch in range(cfg["epochs"]):
        model.train()
        sums, n, skipped = {}, 0, 0
        for imgs, tgts in loader:
            imgs = [im.to(device, non_blocking=True) for im in imgs]
            tgts = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in tgts]
            with torch.autocast("cuda", dtype=torch.float16):
                losses = model(imgs, tgts)
                loss = sum(losses.values())
            if not torch.isfinite(loss):
                skipped += 1
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + float(v.detach())
            n += 1
        log.append({"epoch": epoch + 1, **{k: round(v / max(n, 1), 4) for k, v in sums.items()},
                    "skipped": skipped, "minutes": round((time.time() - t0) / 60, 1)})
        with open(log_path, "w") as f:
            json.dump(log, f)
    torch.save(model.state_dict(), os.path.join(cfg["out_dir"], "model.pt"))
    info = {"train_hours": round((time.time() - t0) / 3600, 2), "epochs": cfg["epochs"], "batch": cfg["batch"],
            "lr": cfg["lr"], "kind": cfg["kind"], "model_kwargs": cfg["model_kwargs"], "n_train": len(cfg["train_paths"])}
    with open(os.path.join(cfg["out_dir"], "info.json"), "w") as f:
        json.dump(info, f)
    return info


def train_parallel(cfgs: list[dict], poll_s: int = 60) -> None:
    """Train configs at the same time, one process per GPU (in waves if there are more configs than GPUs).

    Progress is printed once per finished epoch; a failed run prints its traceback and does not
    stop the others.
    """
    import torch
    import torch.multiprocessing as mp
    ctx = mp.get_context("spawn")
    n_gpu = max(torch.cuda.device_count(), 1)
    for w in range(0, len(cfgs), n_gpu):
        wave = [{**c, "device": f"cuda:{i}"} for i, c in enumerate(cfgs[w:w + n_gpu])]
        procs = [ctx.Process(target=_train_entry, args=(c,)) for c in wave]
        for p in procs:
            p.start()
        seen = {c["name"]: 0 for c in wave}
        while any(p.is_alive() for p in procs):
            time.sleep(poll_s)
            for c in wave:
                lp = os.path.join(c["out_dir"], "log.json")
                if os.path.exists(lp):
                    try:
                        log = json.load(open(lp))
                    except json.JSONDecodeError:
                        continue
                    for row in log[seen[c["name"]]:]:
                        print(c["name"], row, flush=True)
                    seen[c["name"]] = len(log)
        for p, c in zip(procs, wave):
            p.join()
            err = os.path.join(c["out_dir"], "error.txt")
            if p.exitcode != 0 and os.path.exists(err):
                print(f"{c['name']} FAILED:\n{open(err).read()}", flush=True)
            elif p.exitcode == 0:
                print(f"{c['name']} done", flush=True)


def load_detector(kind: str, model_kwargs: dict, weights: str, device: str = "cuda:0"):
    import torch
    model = build_detector(kind, **model_kwargs)
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    return model.to(device).eval()


def detect(model, img_path: str, device: str = "cuda:0"):
    """One image -> (boxes_xyxy, scores, cls 0-based) in original pixels."""
    import torch
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img).permute(2, 0, 1).float().div_(255).to(device)
    with torch.no_grad():
        out = model([t])[0]
    # torchvision label 0 is background; RetinaNet does not filter it itself (Faster R-CNN does)
    keep = out["labels"] > 0
    return (out["boxes"][keep].cpu().numpy(), out["scores"][keep].cpu().numpy(),
            out["labels"][keep].cpu().numpy() - 1)


def rpn_proposals(model, img_path: str, device: str = "cuda:0") -> np.ndarray:
    """Faster R-CNN region proposals (post-NMS, test-time count) in original pixels."""
    import torch
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    h0, w0 = img.shape[:2]
    t = torch.from_numpy(img).permute(2, 0, 1).float().div_(255).to(device)
    with torch.no_grad():
        images, _ = model.transform([t])
        feats = model.backbone(images.tensors)
        props, _ = model.rpn(images, feats)
    hs, ws = images.image_sizes[0]
    p = props[0].cpu().numpy()
    return p * np.array([w0 / ws, h0 / hs, w0 / ws, h0 / hs])
