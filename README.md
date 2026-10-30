# 🏋️ AI-Powered Gym Training Chatbot

A Python virtual fitness assistant that uses **Natural Language Processing** and **supervised Machine Learning** (Scikit-learn) to understand gym-related questions and reply with curated, safe guidance. It has a Flask web interface, a rule-based workout recommender, and an optional SQLite progress log.

> General fitness information only, not medical advice.

---

## ✨ Features

| Feature | Description |
|---|---|
| **Intent recognition** | TF-IDF (word + character n-grams) + Logistic Regression classifies messages into 15 intents |
| **Confidence check** | Predictions below 40% confidence get a polite "please rephrase" instead of a guess |
| **Exercise guide** | 20 exercises with steps, muscles worked, common mistakes and safety notes (entity matching on names and synonyms) |
| **Workout plan builder** | Multi-step conversation (experience → equipment → days) that picks from 7 templates |
| **Fitness FAQ** | Muscle gain, fat loss, recovery, warm-up, hydration, nutrition, motivation |
| **Safety layer** | Rules catch emergencies (chest pain, fainting), injuries/medical conditions and unsafe requests (steroids, crash diets) *before* the ML model |
| **Progress log** | Log workouts, see stats, save plan preferences (SQLite, anonymous ID, delete anytime) |
| **AI details toggle** | Shows predicted intent, confidence and top-3 guesses for every reply (great for demos) |
| **Responsive UI** | Works on desktop and mobile |

---

## 🖥️ System Requirements

- **OS:** Windows 10/11, macOS or Linux
- **Python:** 3.10 or newer (tested on 3.13)
- **RAM:** 4 GB+ · **Disk:** ~500 MB including packages · **GPU:** not needed
- **Internet:** only for installing packages. The bot runs fully offline.

---

## 🚀 Setup & Run

> Full step-by-step guide (Windows/Linux/macOS, troubleshooting, how to extend the model): **[INSTRUCTIONS.md](INSTRUCTIONS.md)**

```bash
# 1. Get the project and open a terminal in its folder
git clone https://github.com/umaiiirkhan/repforge.git
cd repforge

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model (about 5 seconds; prints all evaluation metrics)
python train_model.py

# 5. Start the web app
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

Other commands:

```bash
python -m pytest -v      # run the 52 automated tests
python chatbot.py        # chat in the terminal (no browser needed)
```

---

## 📁 Project Structure

```
AI_Gym_Training_Bot/
├── app.py                 # Flask server & REST API
├── chatbot.py             # Bot logic: ML prediction, safety rules, exercise lookup, workout flow
├── train_model.py         # Training, model comparison, evaluation, saving the pipeline
├── nlp_utils.py           # Text preprocessing (shared by training and prediction)
├── storage.py             # SQLite: workout logs & preferences
├── config.py              # Paths, confidence threshold, settings
├── requirements.txt
├── data/
│   ├── intents.json       # 15 intents, ~900 labelled example sentences + responses
│   ├── exercises.json     # 20 exercises (steps, muscles, mistakes, safety)
│   └── workouts.json      # 7 workout templates
├── models/
│   ├── intent_pipeline.joblib   # trained pipeline (created by train_model.py)
│   └── model_meta.json
├── reports/
│   ├── metrics.json             # all evaluation results + misclassified examples
│   ├── classification_report.txt
│   └── confusion_matrix.png
├── templates/index.html   # Chat UI
├── static/style.css, script.js
└── tests/                 # pytest suite (bot logic + API)
```

---

## 🧠 How It Works

```
User message
   │
   ▼
Input validation (empty / too long)
   │
   ▼
Safety rules ──► emergency / injury / unsafe request → fixed safe reply
   │
   ▼
Active workout-plan conversation? ──► parse answer → next question / final plan
   │
   ▼
Text preprocessing (lowercase, slang & contractions, remove symbols)
   │
   ▼
TF-IDF features: word 1–2 grams + character 2–5 grams (typo-tolerant)
   │
   ▼
Logistic Regression → intent + probability
   │
   ▼
Confidence ≥ 0.40 ? ──no──► "Could you rephrase?" + suggestions
   │yes
   ▼
Exercise name found? → detailed exercise guide
Workout intent?      → start plan builder
Otherwise            → curated response from intents.json
```

### Machine learning pipeline
1. Load and validate `intents.json`, then remove duplicate sentences.
2. Stratified 80/20 train/test split. Test sentences that are near-duplicates of training sentences (cosine similarity ≥ 0.85) are moved to training so the score isn't inflated.
3. Compare **Logistic Regression**, **Linear SVM (calibrated)** and **Complement Naive Bayes** with 5-fold cross-validation (macro F1).
4. Evaluate the best model on the held-out test set: accuracy, precision, recall, F1, confusion matrix.
5. Refit on all data and save the whole pipeline (preprocessing + vectorizer + classifier) with `joblib`.

### Results (held-out test set, 176 sentences)

| Metric | Score |
|---|---|
| Selected model | Logistic Regression |
| Cross-validation macro F1 | 0.79 (LR) · 0.78 (SVM) · 0.77 (NB) |
| Test accuracy | **82.4%** |
| Macro precision / recall / F1 | 0.83 / 0.82 / 0.82 |
| Questions answered at ≥ 40% confidence | 83.5% |
| Accuracy on those confident answers | **89.8%** |

The confidence threshold trades coverage for reliability: uncertain questions get a clarification request instead of a possibly wrong answer. Exact numbers can vary slightly between scikit-learn versions; your own run of `train_model.py` prints the current values.

---

## 🔌 API

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/chat` | `{"message": "..."}` → `{reply, intent, confidence, top_intents, quick_replies, source}` |
| `POST` | `/api/reset` | Clear the conversation state |
| `GET` | `/api/logs` | List workout logs + summary |
| `POST` | `/api/logs` | `{"workout", "duration_min", "date", "notes"}` |
| `DELETE` | `/api/logs/<id>` | Delete one log |
| `DELETE` | `/api/data` | Delete all of the user's data |
| `GET` | `/api/health` | Health check |

---

## 🧪 Testing

`python -m pytest -v` runs 52 tests covering every case in the project's testing table:

| Test case | Expected result |
|---|---|
| Greeting | Greeting response |
| Exercise technique question | Matching exercise guidance |
| Beginner workout request | Multi-step plan builder → template |
| Unrelated question | Out-of-scope / fallback response |
| Empty input | Validation message (UI and API return 400) |
| Low-confidence query | Clarification request |
| Repeated requests | Consistent, error-free responses |

Plus safety rules, data-file validation, template selection, storage, privacy (users can't see each other's logs) and a minimum model-quality check.

---

## ✏️ Customising

- **Add or improve intents:** edit `data/intents.json`, then run `python train_model.py`.
- **Add exercises:** add an entry to `data/exercises.json` (no retraining needed).
- **Add workout templates:** edit `data/workouts.json`.
- **Change strictness:** adjust `CONFIDENCE_THRESHOLD` in `config.py`.

## ⚠️ Limitations

- The classifier only knows the 15 intents it was trained on; it does not truly "understand" language like a large language model.
- Responses are general and prewritten; they are not personalised medical, dietary or coaching advice.
- Rule-based safety checks reduce risk but can't catch every possible phrasing.

## 🔮 Future Enhancements

Voice input, multilingual (Urdu) support, exercise videos, a larger dataset, progress charts, and deployment to the cloud.
