import os
import sys
import importlib.util
from flask import Flask, render_template
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from dotenv import load_dotenv

# Ensure shared_utils (plain folder) is importable without packaging
_PORTAL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PORTAL_ROOT not in sys.path:
    sys.path.insert(0, _PORTAL_ROOT)
# Also add repository root so `shared_utils/` at root is importable
_REPO_ROOT = os.path.abspath(os.path.join(_PORTAL_ROOT, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _load_sub_app(module_name: str, file_path: str):
    module_dir = os.path.dirname(file_path)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {module_name} at {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "app"):
        raise RuntimeError(f"模块 {module_name} 未暴露 Flask 变量 'app'")
    return module.app


def create_app() -> Flask:
    load_dotenv()
    app = Flask(__name__, template_folder="templates", static_folder="static")

    @app.route("/")
    def index():
        targets = {
            "mayuan": {
                "name": "马原助手",
                "desc": "马克思主义基本原理智能学习与问答",
                "url": "/mayuan/",
            },
            "jindaishi": {
                "name": "史纲助手",
                "desc": "中国近现代史纲要智能学习与问答",
                "url": "/jindaishi/",
            },
            "sdfz": {
                "name": "思修法治助手",
                "desc": "思想道德与法治智能学习与问答",
                "url": "/sdfz/",
            },
                "maogai": {
                    "name": "毛概助手",
                    "desc": "毛泽东思想概论智能学习与问答",
                    "url": "/maogai/",
                },
                "xigai": {
                    "name": "习概助手",
                    "desc": "习近平新时代中国特色社会主义思想概论智能学习与问答",
                    "url": "/xigai/",
                },
        }
        return render_template("index.html", targets=targets)

    @app.route("/healthz")
    def healthz():
        return {"status": "ok"}

    # Mount 3 sub-apps under one process
    # Resolve workspace root robustly (support nested folder named the same)
    candidate_roots = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)),
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)),
    ]
    base_dir = None
    for root in candidate_roots:
        if os.path.isdir(os.path.join(root, "A-assistant-to-the-basic-principles-of-Marxism-main")):
            base_dir = root
            break
    if base_dir is None:
        # fallback to parent of current file
        base_dir = candidate_roots[0]

    mayuan_path = os.path.join(base_dir, "A-assistant-to-the-basic-principles-of-Marxism-main", "app.py")
    jindaishi_path = os.path.join(base_dir, "B-assistant-to-the-outline-of-modern-chinese-history-main", "app.py")
    sdfz_path = os.path.join(base_dir, "C-assistant-to-ideology-and-rule-of-law-main", "app.py")
    maogai_path = os.path.join(base_dir, "D-assistant-to-the-introduction-of-mao-zedong-thought-main", "app.py")
    xigai_path = os.path.join(base_dir, "E-assistant-to-the-introduction-of-xi-jinping-thought-main", "app.py")

    mayuan_app = _load_sub_app("mayuan_app", mayuan_path)
    jindaishi_app = _load_sub_app("jindaishi_app", jindaishi_path)
    sdfz_app = _load_sub_app("sdfz_app", sdfz_path)
    maogai_app = _load_sub_app("maogai_app", maogai_path)
    xigai_app = _load_sub_app("xigai_app", xigai_path)

    app.wsgi_app = DispatcherMiddleware(
        app.wsgi_app,
        {
            "/mayuan": mayuan_app,
            "/jindaishi": jindaishi_app,
            "/sdfz": sdfz_app,
            "/maogai": maogai_app,
            "/xigai": xigai_app,
        },
    )

    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)


