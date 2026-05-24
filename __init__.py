"""Flask 应用工厂 - 无人机检测"""
import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from config import Config

db = SQLAlchemy()


def create_app() -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config.from_object(Config)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["RESULT_FOLDER"], exist_ok=True)

    db.init_app(app)

    from app.views import main_bp, api_bp
    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp)

    with app.app_context():
        from app import models  # noqa
        db.create_all()
        models.bootstrap_seed()

    return app
