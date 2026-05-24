"""视图与 API"""
import os
import json
import uuid
from flask import (Blueprint, render_template, request, current_app,
                   jsonify, url_for)
from werkzeug.utils import secure_filename

from app import db
from app.models import Experiment, DetectRecord
from app.detector import run_inference, CLASS_CN

main_bp = Blueprint("main", __name__)
api_bp = Blueprint("api", __name__, url_prefix="/api")


def _allowed(name):
    return "." in name and name.rsplit(".", 1)[1].lower() in current_app.config["ALLOWED_EXT"]


def _visible_experiments():
    """只展示真实训练的「本文方法」一组（P2 + CBAM + WIoU 全启用）。
    其他 7 行占位实验留在 DB 里不删，便于后续真实跑完再放开展示。"""
    rows = (Experiment.query
            .filter(Experiment.use_p2.is_(True),
                    Experiment.use_cbam.is_(True),
                    Experiment.use_wiou.is_(True))
            .order_by(Experiment.id).all())
    if not rows:
        # 兜底：DB 里居然没有「全开」那组，退化展示全部
        rows = Experiment.query.order_by(Experiment.id).all()
    return rows


@main_bp.route("/")
def index():
    exps = [e.to_dict() for e in _visible_experiments()]
    headline = exps[0] if exps else None
    return render_template("index.html", headline=headline, exp_count=len(exps))


@main_bp.route("/dashboard")
def dashboard():
    exps = [e.to_dict() for e in _visible_experiments()]
    return render_template("dashboard.html", experiments=exps)


@main_bp.route("/detect")
def detect_page():
    exps = [e.to_dict() for e in _visible_experiments()]
    return render_template("detect.html", experiments=exps)


@main_bp.route("/ablation")
def ablation():
    exps = [e.to_dict() for e in _visible_experiments()]
    return render_template("ablation.html", experiments=exps)


@main_bp.route("/records")
def records():
    page = int(request.args.get("page", 1))
    per = 12
    pagination = DetectRecord.query.order_by(DetectRecord.created_at.desc()).paginate(
        page=page, per_page=per, error_out=False
    )
    items = [r.to_dict() for r in pagination.items]
    return render_template("records.html", items=items, page=page, pages=pagination.pages)


@api_bp.route("/detect", methods=["POST"])
def api_detect():
    file = request.files.get("file")
    exp_id = request.form.get("experiment_id")
    if not file or not exp_id:
        return jsonify({"success": False, "message": "缺少图像或实验配置"}), 400
    if not _allowed(file.filename):
        return jsonify({"success": False, "message": "文件类型不支持"}), 400
    exp = Experiment.query.get(int(exp_id))
    if not exp:
        return jsonify({"success": False, "message": "实验不存在"}), 404

    safe = secure_filename(file.filename)
    name = f"{uuid.uuid4().hex[:8]}_{safe}"
    upload_path = os.path.join(current_app.config["UPLOAD_FOLDER"], name)
    file.save(upload_path)

    try:
        out = run_inference(upload_path, current_app.config["RESULT_FOLDER"], exp)
    except Exception as e:
        current_app.logger.error("inference failed", exc_info=True)
        return jsonify({"success": False, "message": str(e)}), 500

    rec = DetectRecord(
        image_path=os.path.relpath(upload_path, current_app.static_folder).replace("\\", "/"),
        result_path=os.path.relpath(out["save_path"], current_app.static_folder).replace("\\", "/"),
        experiment_id=exp.id,
        summary=out["summary"],
        objects=json.dumps(out["items"], ensure_ascii=False),
        inference_ms=out["inference_ms"],
    )
    db.session.add(rec)
    db.session.commit()

    counts = {}
    for it in out["items"]:
        counts[it["label"]] = counts.get(it["label"], 0) + 1
    cn_counts = [{"name": CLASS_CN.get(k, k), "value": v} for k, v in counts.items()]

    return jsonify({"success": True, "data": {
        "record": rec.to_dict(),
        "image_url": url_for("static", filename=rec.image_path),
        "result_url": url_for("static", filename=rec.result_path),
        "class_distribution": cn_counts,
    }, "message": "完成"})


@api_bp.route("/experiments")
def api_experiments():
    items = [e.to_dict() for e in _visible_experiments()]
    return jsonify({"success": True, "data": items})
