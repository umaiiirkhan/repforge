"""Flask web server for the AI Gym Training Bot.

Run:  python app.py   then open http://127.0.0.1:5000
"""
import os
import re
import secrets
import uuid
from datetime import date

from flask import Flask, jsonify, render_template, request, session

import config
from chatbot import GymBot
from storage import Storage


def create_app(test_config=None):
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=os.environ.get("SECRET_KEY") or secrets.token_hex(32),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        JSON_SORT_KEYS=False,
        DB_PATH=config.DB_PATH,
    )
    if test_config:
        app.config.update(test_config)

    if not config.MODEL_PATH.exists():
        print("No trained model found, training it now (takes a few seconds) ...")
        import train_model
        train_model.main()

    storage = Storage(app.config["DB_PATH"])
    bot = GymBot(storage=storage)

    def current_user_id():
        """Anonymous random id stored in the session cookie (no login or personal data)."""
        if "user_id" not in session:
            session["user_id"] = uuid.uuid4().hex
        return session["user_id"]

    def error(message, status=400):
        return jsonify({"error": message}), status

    # ----------------------------------------------------------------- pages
    @app.get("/")
    def index():
        current_user_id()
        return render_template("index.html")

    # ------------------------------------------------------------------ chat
    @app.post("/api/chat")
    def chat():
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or not isinstance(data.get("message"), str):
            return error("Request body must be JSON like {\"message\": \"...\"}")
        message = data["message"].strip()
        if not message:
            return error("Message cannot be empty.")
        if len(message) > config.MAX_MESSAGE_LENGTH:
            return error(f"Message must be at most {config.MAX_MESSAGE_LENGTH} characters.")

        user_id = current_user_id()
        result, new_state = bot.respond(message, session.get("chat_state", {}), user_id=user_id)
        session["chat_state"] = new_state
        return jsonify(result)

    @app.post("/api/reset")
    def reset():
        session.pop("chat_state", None)
        return jsonify({"ok": True})

    # ---------------------------------------------------------- progress log
    @app.get("/api/logs")
    def list_logs():
        uid = current_user_id()
        return jsonify({"logs": storage.get_logs(uid), "summary": storage.summary(uid)})

    @app.post("/api/logs")
    def add_log():
        data = request.get_json(silent=True) or {}
        workout = str(data.get("workout", "")).strip()
        notes = str(data.get("notes", "")).strip()
        log_date = str(data.get("date") or date.today().isoformat())
        try:
            duration = int(data.get("duration_min", 0))
        except (TypeError, ValueError):
            return error("Duration must be a whole number of minutes.")
        if not workout or len(workout) > 80:
            return error("Workout name is required (max 80 characters).")
        if not 1 <= duration <= 600:
            return error("Duration must be between 1 and 600 minutes.")
        if len(notes) > 300:
            return error("Notes must be at most 300 characters.")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", log_date):
            return error("Date must be in YYYY-MM-DD format.")
        try:
            if date.fromisoformat(log_date) > date.today():
                return error("Date cannot be in the future.")
        except ValueError:
            return error("Invalid date.")
        log_id = storage.add_log(current_user_id(), workout, duration, log_date, notes)
        return jsonify({"ok": True, "id": log_id}), 201

    @app.delete("/api/logs/<int:log_id>")
    def delete_log(log_id):
        if not storage.delete_log(current_user_id(), log_id):
            return error("Log not found.", 404)
        return jsonify({"ok": True})

    @app.delete("/api/data")
    def delete_all_data():
        storage.delete_user_data(current_user_id())
        session.pop("chat_state", None)
        return jsonify({"ok": True})

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok"})

    @app.errorhandler(404)
    def not_found(_):
        if request.path.startswith("/api/"):
            return error("Not found.", 404)
        return render_template("index.html"), 404

    @app.errorhandler(405)
    def method_not_allowed(_):
        return error("Method not allowed.", 405)

    @app.errorhandler(500)
    def server_error(_):
        return error("Something went wrong on the server.", 500)

    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG") == "1"
    print(f"\n  AI Gym Training Bot running at  http://127.0.0.1:{port}\n")
    create_app().run(host="127.0.0.1", port=port, debug=debug)
