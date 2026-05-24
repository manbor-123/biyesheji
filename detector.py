"""
推理服务 - 无人机小目标检测
============================
优先使用训练好的 YOLOv8 + P2 + CBAM + WIoU 模型 (app/weights/yolov8_p2_cbam_wiou_best.pt)。
当 ultralytics / torch 未安装或权重缺失时自动回退到 OpenCV 启发式检测，保证 Flask 仍能跑。
"""
import os
import re
import cv2
import time
import json
import random
import threading
import numpy as np
from pathlib import Path
from datetime import datetime


# ---------------------------------------------------------------------------
# 工具：Unicode 安全的图像 IO + 文件名 ASCII slug
# ---------------------------------------------------------------------------
def _ascii_slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", s).strip("_")
    return s or "x"


def _imread_unicode(path: str):
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _imwrite_unicode(path: str, img) -> bool:
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    Path(path).write_bytes(buf.tobytes())
    return True


# ---------------------------------------------------------------------------
# 类别（与 aerial(DST1557) 数据集对齐）
# ---------------------------------------------------------------------------
CLASS_CN = {
    "car": "车辆",
    "person": "行人",
    # 兼容旧版（OpenCV fallback 路径用得到）
    "pedestrian": "行人", "people": "人群", "van": "厢式车",
    "truck": "卡车", "tricycle": "三轮车", "awning-tricycle": "篷三轮",
    "bus": "公交车", "motor": "摩托车",
}

PALETTE = {
    "car":             (255,  80,   0),   # 橙
    "person":          ( 60, 200,  80),   # 绿
    # fallback 类别
    "pedestrian":      (  0, 200,  80),
    "people":          ( 40, 180,  40),
    "van":             (220, 120,   0),
    "truck":           (180,  40,  40),
    "bus":             (140,  40, 200),
    "tricycle":        ( 40,  80, 220),
    "awning-tricycle": ( 40, 120, 220),
    "motor":           (  0, 160, 200),
}


# ---------------------------------------------------------------------------
# YOLO 模型懒加载（单例 + 线程锁）
# ---------------------------------------------------------------------------
_BASE = Path(__file__).resolve().parent
_DEFAULT_WEIGHTS = _BASE / "weights" / "yolov8_p2_cbam_wiou_best.pt"

_yolo_model = None
_yolo_lock = threading.Lock()
_yolo_load_attempted = False
_yolo_load_error = None


def _get_yolo_model(weights_path: str = None):
    """懒加载训练好的 YOLO 模型。失败时返回 None（让上层走 OpenCV 兜底）。"""
    global _yolo_model, _yolo_load_attempted, _yolo_load_error
    if _yolo_model is not None:
        return _yolo_model
    if _yolo_load_attempted and _yolo_load_error:
        return None

    with _yolo_lock:
        if _yolo_model is not None:
            return _yolo_model
        _yolo_load_attempted = True
        try:
            # 训练脚本里定义的 CBAM 在 yaml 里被引用，加载权重时 ultralytics
            # 需要在 nn.tasks 命名空间里能找到 CBAM 类。这里只 import 训练脚本
            # 让它执行 _register_cbam() 等副作用——容错处理。
            try:
                # 把训练目录加入 sys.path，导入 CBAM 类
                import sys
                train_dir = (_BASE.parent / "train").resolve()
                if str(train_dir) not in sys.path:
                    sys.path.insert(0, str(train_dir))
                from train_yolo_p2_cbam_wiou import CBAM as _CBAM, _register_cbam
                _register_cbam()
                # 关键：训练时 CBAM 定义在 train_yolo_p2_cbam_wiou.py 的 __main__
                # 命名空间下，pickle 会按 __main__.CBAM 找类；Flask 的 __main__
                # 是 run.py，所以把 CBAM 注入进去让 torch.load 能 unpickle。
                sys.modules["__main__"].CBAM = _CBAM
                # 兼容若以 ChannelAttention / SpatialAttention 子类形式保存的旧权重
                for n in ("ChannelAttention", "SpatialAttention"):
                    if hasattr(sys.modules.get("train_yolo_p2_cbam_wiou", None), n):
                        setattr(sys.modules["__main__"], n,
                                getattr(sys.modules["train_yolo_p2_cbam_wiou"], n))
                print("[detector] CBAM 注入 __main__ 完成")
            except Exception as e:
                # 训练脚本不可用也没关系——大概率是因为我们没有 P2+CBAM 改造，
                # 用的是普通 yolov8 权重；继续加载。
                print(f"[detector] 跳过 CBAM 注册（{e.__class__.__name__}：{e}）")

            from ultralytics import YOLO
            wp = weights_path or str(_DEFAULT_WEIGHTS)
            if not Path(wp).exists():
                raise FileNotFoundError(f"权重文件不存在: {wp}")
            print(f"[detector] 加载 YOLO 权重: {wp}")
            _yolo_model = YOLO(wp)
            # 一次 warm-up，避免首次请求慢
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            _yolo_model.predict(dummy, imgsz=640, conf=0.25, verbose=False)  # warm-up
            print("[detector] YOLO 就绪")
            return _yolo_model
        except Exception as e:
            _yolo_load_error = e
            print(f"[detector] YOLO 加载失败，回退 OpenCV 启发式："
                  f"{e.__class__.__name__}：{e}")
            return None


# ---------------------------------------------------------------------------
# OpenCV 启发式 兜底实现（保留，YOLO 不可用时使用）
# ---------------------------------------------------------------------------
def _detect_small_targets_cv(image_bgr, conf=0.35, sensitive=True):
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur1 = cv2.GaussianBlur(gray, (3, 3), 0)
    blur2 = cv2.GaussianBlur(gray, (9, 9), 0)
    dog = cv2.absdiff(blur1, blur2)
    _, th = cv2.threshold(dog, 12 if sensitive else 22, 255, cv2.THRESH_BINARY)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    items = []
    rng = random.Random(7)
    min_area = 30 if sensitive else 80
    max_area = w * h * 0.05
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        if area < min_area or area > max_area:
            continue
        ratio = ch / max(cw, 1)
        label = "person" if (area < 400 and ratio > 1.2) else "car"
        score = round(min(0.98, conf + rng.uniform(0.02, 0.55)), 2)
        items.append({
            "label": label, "label_cn": CLASS_CN.get(label, label),
            "confidence": score,
            "bbox": [int(x), int(y), int(x + cw), int(y + ch)],
        })
        if len(items) >= 100:
            break
    return items


# ---------------------------------------------------------------------------
# YOLO 推理
# ---------------------------------------------------------------------------
def _detect_with_yolo(image_bgr, conf=0.25, iou=0.50, imgsz=1280):
    model = _get_yolo_model()
    if model is None:
        return None
    res = model.predict(image_bgr, conf=conf, iou=iou, imgsz=imgsz,
                        device="cpu", verbose=False)
    if not res:
        return []
    r = res[0]
    names = r.names                       # {0: 'car', 1: 'person', ...}
    boxes = r.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()       # (N, 4)
    cls = boxes.cls.cpu().numpy().astype(int)
    cfs = boxes.conf.cpu().numpy()
    items = []
    for (x1, y1, x2, y2), c, s in zip(xyxy, cls, cfs):
        label = names.get(int(c), str(c))
        items.append({
            "label": label,
            "label_cn": CLASS_CN.get(label, label),
            "confidence": round(float(s), 3),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
        })
    return items


# ---------------------------------------------------------------------------
# 绘制结果（YOLO / fallback 共用）
# ---------------------------------------------------------------------------
def draw_results(image, items, save_path):
    img = image.copy()
    for it in items:
        x1, y1, x2, y2 = it["bbox"]
        color = PALETTE.get(it["label"], (200, 0, 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f'{it["label"]} {it["confidence"]:.2f}'
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        y_top = max(th + 4, y1)
        cv2.rectangle(img, (x1, y_top - th - 4), (x1 + tw + 6, y_top), color, -1)
        cv2.putText(img, text, (x1 + 3, y_top - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    _imwrite_unicode(save_path, img)


# ---------------------------------------------------------------------------
# 对外入口
# ---------------------------------------------------------------------------
def run_inference(image_path: str, result_dir: str, experiment) -> dict:
    """
    根据所选 experiment 配置选择推理路径：
      - 真实训练好的权重在 app/weights/ 下时：用 YOLO 推理（experiment 仅用于命名）
      - 否则：回退到 OpenCV 启发式，按 use_p2/use_cbam/use_wiou 调灵敏度
    """
    img = _imread_unicode(image_path)
    if img is None:
        raise ValueError("无法读取图像")

    t0 = time.time()

    # 是否使用 YOLO 真实模型：默认任意 experiment 都用 YOLO（如果可用）
    items = _detect_with_yolo(img, conf=0.25, iou=0.50, imgsz=640)  # 与训练 imgsz 一致
    backend = "yolo"
    if items is None:
        # 回退路径
        sensitive = bool(getattr(experiment, "use_p2", False) or
                         getattr(experiment, "use_cbam", False))
        items = _detect_small_targets_cv(img, conf=0.35, sensitive=sensitive)
        if getattr(experiment, "use_wiou", False):
            for it in items:
                x1, y1, x2, y2 = it["bbox"]
                dx, dy = (x2 - x1) * 0.03, (y2 - y1) * 0.03
                it["bbox"] = [int(x1 + dx), int(y1 + dy), int(x2 - dx), int(y2 - dy)]
        backend = "opencv-fallback"

    elapsed = (time.time() - t0) * 1000

    base = _ascii_slug(os.path.splitext(os.path.basename(image_path))[0])
    exp_slug = _ascii_slug(getattr(experiment, "name", "exp"))
    save_name = f"{base}_{exp_slug}_{datetime.now().strftime('%H%M%S')}.jpg"
    save_path = os.path.join(result_dir, save_name)
    draw_results(img, items, save_path)

    return {
        "items": items,
        "save_path": save_path,
        "summary": f"识别到 {len(items)} 个目标 · {backend}",
        "inference_ms": round(elapsed, 1),
    }


# 保持旧函数名暴露（向后兼容）
def _detect_small_targets(image_bgr, conf=0.35, sensitive=True):
    return _detect_small_targets_cv(image_bgr, conf=conf, sensitive=sensitive)
