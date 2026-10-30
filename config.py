"""Central configuration for the AI Gym Training Bot."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

INTENTS_PATH = DATA_DIR / "intents.json"
EXERCISES_PATH = DATA_DIR / "exercises.json"
WORKOUTS_PATH = DATA_DIR / "workouts.json"
MODEL_PATH = MODEL_DIR / "intent_pipeline.joblib"
MODEL_META_PATH = MODEL_DIR / "model_meta.json"
DB_PATH = DATA_DIR / "gym_bot.db"

# Predictions below this probability trigger a "please rephrase" fallback.
CONFIDENCE_THRESHOLD = 0.40

# Input validation
MAX_MESSAGE_LENGTH = 500

# Training
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
NEAR_DUPLICATE_SIMILARITY = 0.85
