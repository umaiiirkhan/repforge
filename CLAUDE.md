# Project context for Claude Code

College AI project: AI-Powered Gym Training Chatbot (Python, Scikit-learn, Flask).

## Commands
- Install: `pip install -r requirements.txt`
- Train: `python train_model.py` (rerun after ANY change to data/intents.json or nlp_utils.py)
- Run: `python app.py` → http://127.0.0.1:5000
- Test: `python -m pytest -v` (all tests must pass)

## Architecture
- `chatbot.py` GymBot.respond(message, state, user_id) → (response_dict, new_state). Order: validation → safety rules → storage commands → workout flow → exercise entity → ML intent → confidence threshold → response.
- `nlp_utils.normalize` is used INSIDE the saved pipeline; changing it requires retraining.
- Conversation state lives in the Flask session cookie (keep it small).
- Data files: data/intents.json (training data + responses), data/exercises.json, data/workouts.json.

## Rules
- Keep fitness advice general and safe; never add medical diagnosis, extreme diets or drug advice.
- Every intent should keep at least 25 varied example patterns; never put the same sentence in two intents.
- After changing data, retrain and report the new accuracy / macro F1 from reports/metrics.json.
