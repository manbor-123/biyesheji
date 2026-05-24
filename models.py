"""数据模型 - 实验记录、检测结果、消融实验"""
from datetime import datetime
import json
from app import db


def _safe(v):
    if isinstance(v, datetime):
        return v.strftime("%Y-%m-%d %H:%M:%S")
    return v


class Experiment(db.Model):
    """模型/消融实验配置与指标"""
    __tablename__ = "experiment"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), unique=True, nullable=False)
    backbone = db.Column(db.String(32), default="YOLOv8n")
    use_p2 = db.Column(db.Boolean, default=False)
    use_cbam = db.Column(db.Boolean, default=False)
    use_wiou = db.Column(db.Boolean, default=False)
    map50 = db.Column(db.Float, default=0.0)
    map5095 = db.Column(db.Float, default=0.0)
    precision = db.Column("precision_", db.Float, default=0.0)  # DB 列名是 precision_，因为 PRECISION 是 SQL 保留字
    recall = db.Column(db.Float, default=0.0)
    fps = db.Column(db.Float, default=0.0)
    note = db.Column(db.String(255), default="")
    created_at = db.Column(db.DateTime, default=datetime.now)

    def to_dict(self):
        return {
            "id": self.id, "name": self.name, "backbone": self.backbone,
            "use_p2": bool(self.use_p2), "use_cbam": bool(self.use_cbam),
            "use_wiou": bool(self.use_wiou),
            "map50": float(self.map50), "map5095": float(self.map5095),
            "precision": float(self.precision), "recall": float(self.recall),
            "fps": float(self.fps), "note": self.note,
            "created_at": _safe(self.created_at),
        }


class DetectRecord(db.Model):
    """单次推理记录"""
    __tablename__ = "detect_record"
    id = db.Column(db.Integer, primary_key=True)
    image_path = db.Column(db.String(255))
    result_path = db.Column(db.String(255))
    experiment_id = db.Column(db.Integer, db.ForeignKey("experiment.id"))
    summary = db.Column(db.String(255))
    objects = db.Column(db.Text)  # JSON 列表
    inference_ms = db.Column(db.Float, default=0.0)
    created_at = db.Column(db.DateTime, default=datetime.now, index=True)

    def to_dict(self):
        try:
            objs = json.loads(self.objects) if self.objects else []
        except Exception:
            objs = []
        return {
            "id": self.id, "image_path": self.image_path,
            "result_path": self.result_path,
            "experiment_id": self.experiment_id,
            "summary": self.summary, "objects": objs,
            "inference_ms": float(self.inference_ms),
            "created_at": _safe(self.created_at),
        }


def bootstrap_seed():
    """初始化 8 组消融实验数据。"""
    if Experiment.query.count() > 0:
        return
    # 来自典型 VisDrone 论文实验的合理估计值（用于 demo）
    seeds = [
        ("Baseline_v8n",          False, False, False, 0.328, 0.183, 0.461, 0.366, 142.0, "YOLOv8n 基线"),
        ("Baseline+P2",           True,  False, False, 0.376, 0.214, 0.498, 0.402, 121.0, "新增 P2 高分辨率检测头"),
        ("Baseline+CBAM",         False, True,  False, 0.354, 0.198, 0.482, 0.388, 132.0, "C2f 后嵌入 CBAM 注意力"),
        ("Baseline+WIoU",         False, False, True,  0.349, 0.196, 0.476, 0.391, 140.0, "替换 CIoU -> WIoU"),
        ("P2+CBAM",               True,  True,  False, 0.392, 0.228, 0.512, 0.418, 115.0, "P2 + CBAM"),
        ("P2+WIoU",               True,  False, True,  0.388, 0.224, 0.508, 0.413, 119.0, "P2 + WIoU"),
        ("CBAM+WIoU",             False, True,  True,  0.371, 0.211, 0.495, 0.401, 130.0, "CBAM + WIoU"),
        ("P2+CBAM+WIoU(本文)",    True,  True,  True,  0.418, 0.249, 0.541, 0.444, 113.0, "三者融合,论文提出方法"),
    ]
    for s in seeds:
        e = Experiment(name=s[0], use_p2=s[1], use_cbam=s[2], use_wiou=s[3],
                       map50=s[4], map5095=s[5], precision=s[6], recall=s[7],
                       fps=s[8], note=s[9])
        db.session.add(e)
    db.session.commit()
