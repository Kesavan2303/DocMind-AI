import os
from flask import Flask
from config import Config
from app.extensions import db, login_manager, csrf
from utils.langsmith_setup import setup_langsmith


def create_app() -> Flask:
    setup_langsmith()

    app = Flask(__name__, template_folder="templates", static_folder="../static")
    app.config.from_object(Config)

    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "data"), exist_ok=True)

    db.init_app(app)
    login_manager.init_app(app)
    csrf.init_app(app)

    login_manager.login_view = "auth.login"
    login_manager.login_message = "Please log in to access DocMind AI."
    login_manager.login_message_category = "info"

    from app.models import User

    @login_manager.user_loader
    def load_user(user_id: str):
        return User.query.get(int(user_id))

    # Register blueprints
    from app.auth.routes import auth
    from app.dashboard.routes import dashboard
    from app.documents.routes import documents
    from app.rag.routes import rag

    app.register_blueprint(auth, url_prefix="/auth")
    app.register_blueprint(dashboard, url_prefix="/")
    app.register_blueprint(documents, url_prefix="/documents")
    app.register_blueprint(rag, url_prefix="/rag")

    with app.app_context():
        db.create_all()

    return app
