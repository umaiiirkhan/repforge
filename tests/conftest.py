import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def ensure_model():
    """Train the model once if it hasn't been trained yet."""
    if not config.MODEL_PATH.exists():
        import train_model
        train_model.main()


@pytest.fixture
def storage(tmp_path):
    from storage import Storage
    return Storage(tmp_path / "test.db")


@pytest.fixture
def bot(storage):
    from chatbot import GymBot
    return GymBot(storage=storage)


@pytest.fixture
def client(tmp_path):
    from app import create_app
    app = create_app({"TESTING": True, "DB_PATH": tmp_path / "app.db", "SECRET_KEY": "test"})
    return app.test_client()
