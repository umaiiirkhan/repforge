# RepForge Assistant: Instructions

An AI gym-training chatbot built with **Python, Scikit-learn and Flask**. It answers questions about
exercise form, builds workout plans, gives general recovery and nutrition tips, and keeps a simple
workout log.

---

## 1. Setup and run

**Prerequisites:** Python 3.10 or newer, plus internet for the first install.
You do *not* need Node.js, a database server or anything else. The database is SQLite, which comes with Python.

**Linux / macOS**
```bash
sudo apt install python3-venv          # Ubuntu/Debian only, once
cd AI_Gym_Training_Bot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_model.py                  # creates models/intent_pipeline.joblib
python -m pytest -q                    # all tests should pass
python app.py                          # open http://127.0.0.1:5000
```

**Windows (PowerShell)**
```powershell
cd AI_Gym_Training_Bot
py -m venv .venv
.venv\Scripts\Activate.ps1             # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
python train_model.py
python -m pytest -q
python app.py
```

Stop the server with **Ctrl+C**. Next time, just activate the venv and run `python app.py`.

---

## 2. Working on the project (short summary)

### Project map
| Path | What it is |
|---|---|
| `app.py` | Flask web server and API routes (`/api/chat`, `/api/logs`, …) |
| `chatbot.py` | The bot's brain: `GymBot.respond()` decides every reply |
| `nlp_utils.py` | Text cleaning (`normalize`), used inside the trained model |
| `train_model.py` | Trains and compares 3 models, saves the best one and writes reports |
| `storage.py` | SQLite storage for workout logs and saved preferences |
| `config.py` | Paths, confidence threshold (`0.40`), training settings |
| `data/intents.json` | **Training data**: example questions + answers for each topic |
| `data/exercises.json` | Exercise guides (steps, mistakes, safety) |
| `data/workouts.json` | Workout plan templates |
| `templates/index.html`, `static/` | Web interface (HTML, CSS, JavaScript) |
| `tests/` | Automated tests (pytest) |
| `reports/` | Accuracy metrics, classification report, confusion matrix |

### How a message is answered
`chatbot.py → GymBot.respond()` checks things in this order:

1. **Validation**: empty or too-long message
2. **Safety rules**: chest pain, injuries, steroids, extreme diets (regex rules always win)
3. **Storage commands**: "save my preferences", "delete my data", "show my progress"
4. **Workout-plan flow**: asks experience → equipment → days, then builds a plan
5. **Exercise detection**: finds an exercise name such as "squat" and shows its guide
6. **ML model**: predicts the topic (intent) of the message
7. **Confidence check**: below 0.40 the bot asks you to rephrase
8. **Reply**: a random answer from that intent's `responses`

### Golden rules
- **Retrain after ANY change** to `data/intents.json` or `nlp_utils.py`: `python train_model.py`
- **Run the tests** before sharing: `python -m pytest -q` (all must pass).
- Keep advice **general and safe**. No medical diagnosis, extreme diets or drug/supplement-cycle advice.
- Each intent needs **at least 25 varied patterns**, and the **same sentence must never be in two intents**.
  The tests check both.
- Conversation state lives in the browser session cookie, so keep it small.

---

## 3. Adding more things to the model

### A. Teach an existing topic more phrasings (most common)
Open `data/intents.json`, find the intent by its `tag`, and add sentences to `patterns`:

```json
{
  "tag": "hydration",
  "patterns": [
    "how much water should I drink",
    "is it ok to drink water during a workout",
    "do I need electrolytes after training"
  ],
  "responses": [ "..." ]
}
```
Tips:
- Write how real users type: short, long, casual, with typos ("how much watr").
- Vary the wording; don't just swap one word.
- Add sentences to the intents that score lowest in `reports/classification_report.txt`
  (currently `motivation` and `out_of_scope`).

Then retrain (`python train_model.py`) and check the new scores.

### B. Add a brand-new topic (intent)
1. Add a new block to the `intents` list in `data/intents.json`:
   ```json
   {
     "tag": "sleep",
     "patterns": [
       "how much sleep do I need for muscle growth",
       "does sleep affect my gains",
       "... at least 25 different examples ..."
     ],
     "responses": [
       "Most adults do best with **7–9 hours** of sleep. Muscles repair while you rest, so ...",
       "A second, differently worded answer ..."
     ]
   }
   ```
   - `tag`: lowercase_with_underscores, unique.
   - `patterns`: **25+**, none copied from another intent.
   - `responses`: 1 or more. Supports `**bold**`, `*italic*`, `\n` line breaks, `•` bullets and icon tags (see E).
2. *(Optional)* In `chatbot.py`:
   - add quick-reply buttons for it in the `quick = {...}` dict inside `respond()`
   - add a "did you mean" label in `DID_YOU_MEAN_LABELS`
3. Retrain, then run the tests.

> Adding topics only makes sense for questions the bot should answer **with a fixed text**.
> Anything that needs logic (asking follow-up questions, reading data) must be written in `chatbot.py`.

### C. Add a new exercise guide
Add an object to `data/exercises.json → "exercises"`. **No retraining needed**; restart the app.

```json
{
  "id": "farmer_carry",
  "name": "Farmer's Carry",
  "synonyms": ["farmer carry", "farmers carry", "farmer's walk", "farmers walk"],
  "muscles": ["grip", "traps", "core"],
  "equipment": "dumbbells or kettlebells",
  "level": "beginner",
  "steps": ["Pick up a heavy weight in each hand.", "Stand tall, shoulders back.", "Walk with short, controlled steps."],
  "mistakes": ["Leaning to one side", "Shrugging the shoulders up"],
  "safety": "Start light and put the weights down if your grip or lower back gives out."
}
```
- `synonyms` are how users will name it. Include plural, hyphen and no-space versions.
  Longer synonyms win, so "goblet squat" beats "squat".
- Required fields (checked by tests): `id, name, synonyms, muscles, steps, mistakes, safety`.

### D. Add or change a workout plan
Edit `data/workouts.json → "templates"`. **No retraining needed.**

```json
{
  "id": "dumbbell_push_pull_legs",
  "name": "Dumbbell Push/Pull/Legs Plan",
  "equipment": "dumbbells",
  "split": "ppl",
  "levels": ["intermediate"],
  "sessions": [
    { "focus": "Push", "exercises": ["Dumbbell Bench Press: 3 × 8–12", "..."] },
    { "focus": "Pull", "exercises": ["..."] },
    { "focus": "Legs", "exercises": ["..."] }
  ]
}
```
- `equipment` must be one of `none`, `dumbbells`, `gym`.
- `split` must be one of `full_body`, `upper_lower`, `ppl`.
- Which split is picked (by days per week and experience) is decided in `chatbot.py → select_template`.
  A new combination like the dumbbell PPL above only gets used
  if you update that logic too.
- New equipment types or questions need changes to the parsers in `chatbot.py`.

### E. Icons in bot replies
Write `{icon:NAME}` in any reply text. Available names:
`alert, heart, shield, dumbbell, note, smile, wave, trash, zap, help, save, book, trending, activity, calendar, apple, user, x`.
Inside a Python **f-string**, write it with double braces: `f"{{icon:zap}} Nice work, {name}!"`.
To add a new icon, add a `<symbol id="i-NAME">` to the sprite at the top of `templates/index.html`.

### F. After any data change: checklist
```bash
python train_model.py      # only needed for intents.json / nlp_utils.py changes
python -m pytest -q        # all tests must pass
python app.py              # try your new questions in the browser
```
Then check `reports/metrics.json`:
- `test_set.accuracy` must stay **≥ 0.75** and `accuracy_when_confident` **≥ 0.85** (the tests check this).
- Current baseline: accuracy **0.824**, macro F1 **0.823**.
- Turn on **AI details** in the app header to see the predicted intent and confidence for each reply.
  This helps you spot which topic needs more examples.

---

## 4. Troubleshooting
| Problem | Fix |
|---|---|
| `ensurepip is not available` when creating the venv | `sudo apt install python3-venv` (Ubuntu/Debian) |
| First start is slow / prints training output | No model was found, so `app.py` trains one automatically. Or run `python train_model.py` first |
| `Model not found` error in tests or scripts | Run `python train_model.py` |
| Changed `intents.json` but the bot doesn't use it | Retrain, then restart `python app.py` |
| Changed HTML but the page looks the same | Restart the server, then hard-refresh the browser (Ctrl+Shift+R) |
| Test `no_sentence_used_for_two_intents` fails | The same sentence is in two intents. Remove one copy |
| Test `intents_file_is_valid` fails | An intent has fewer than 25 patterns or no responses |
| `Address already in use` | Another copy of the app is running. Close it, or stop the process using port 5000 |
