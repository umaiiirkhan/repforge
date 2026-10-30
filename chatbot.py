"""Core chatbot logic: intent prediction, exercise lookup, safety rules and the
multi-step workout recommendation flow.

The bot itself is stateless: the caller passes in the conversation `state`
(a small dict) and receives an updated one back. This keeps it easy to test
and lets Flask store the state in the user's session.
"""
import json
import random
import re

import joblib

import config
from nlp_utils import normalize

# ------------------------------------------------------------------ safety
# Checked BEFORE the classifier so these messages are always handled safely.
EMERGENCY_PATTERNS = [
    r"\bchest (pain|hurts?|tightness)\b", r"\bpain in my chest\b", r"\b(can ?not|can't|cant|trouble|difficulty) breath",
    r"\bfaint(ed|ing)?\b", r"\bpass(ed)? out\b", r"\bblack(ed)? out\b", r"\bheart attack\b",
    r"\bheart (is )?(racing|pounding)\b", r"\bpalpitation", r"\bsevere(ly)? dizz",
]
INJURY_PATTERNS = [
    r"\bsharp pain\b", r"\binjur(y|ed|ies)\b", r"\bsprain", r"\btor(e|n)\b", r"\bfracture", r"\bbroken (bone|arm|leg|wrist)\b",
    r"\bswollen\b", r"\bswelling\b", r"\bnumb(ness)?\b", r"\btingling\b", r"\bherniat", r"\bhernia\b", r"\bdislocat",
    r"\b(knee|back|shoulder|wrist|elbow|ankle|hip|neck|joint)s? (pain|hurts?|is hurting)\b",
    r"\b(my|in my) (knee|back|lower back|shoulder|wrist|elbow|ankle|hip|neck|joints?) (hurts?|is hurting|pain)",
    r"\bpain in my (knee|back|lower back|shoulder|wrist|elbow|ankle|hip|neck|joints?)\b",
    r"\bpregnan", r"\bdiabet", r"\bblood pressure\b", r"\basthma\b", r"\bheart (condition|problem|disease)",
    r"\bsurgery\b", r"\bdiagnos",
]
UNSAFE_PATTERNS = [
    r"\bsteroid", r"\bsarms?\b", r"\btestosterone (shot|injection|cycle)", r"\banabolic", r"\bdiuretic",
    r"\blaxative", r"\bstarv", r"\bnot eat(ing)? (for|anything)", r"\bstop eating\b", r"\bskip(ping)? all meals\b",
    r"\b(throw(ing)? up|vomit|purg)", r"\blose \d+ ?(kg|kilos?|pounds|lbs?) in (a|one|1|2|two) (day|days|week)\b",
    r"\beat(ing)? (only )?\d{2,3} calories\b", r"\bfat burner pills?\b", r"\bclenbuterol\b",
]

EMERGENCY_REPLY = (
    "{icon:alert} **Please stop exercising now.** Chest pain, trouble breathing, fainting or a racing heart can be serious.\n"
    "Sit down, rest, and **seek urgent medical help** (call your local emergency number) if symptoms are severe or don't settle quickly.\n\n"
    "I'm a fitness information bot and can't assess medical symptoms."
)
INJURY_REPLY = (
    "{icon:alert} I can't diagnose pain, injuries or medical conditions. Please stop any exercise that causes sharp, joint or worsening pain, "
    "and talk to a **doctor or physiotherapist** before continuing. They can tell you what's safe for your situation.\n\n"
    "Once you're cleared, I'm happy to explain general exercise technique or warm-ups."
)
UNSAFE_REPLY = (
    "I can't help with that. Extreme dieting, very rapid weight loss, purging or performance-enhancing drugs can seriously harm your health.\n\n"
    "Safe progress looks like: a small calorie deficit or surplus, enough protein, regular strength training and good sleep. "
    "For personal guidance, please speak with a doctor or registered dietitian. If you're struggling with food or body image, "
    "talking to someone you trust or a healthcare professional can really help. {icon:heart}"
)

# ------------------------------------------------------------------ misc rules
RESTART_RE = re.compile(r"\b(start over|different plan|new plan|another plan|restart)\b")
CANCEL_RE = re.compile(r"\b(cancel|stop|quit|exit|nevermind|never mind|forget it|start over|reset)\b")
SAVE_PREFS_RE = re.compile(r"\bsave (my )?(preferences|prefs|settings|plan)\b")
DELETE_DATA_RE = re.compile(r"\b(delete|erase|remove|clear) (all )?(my )?(data|information|history|logs|preferences)\b")
PROGRESS_RE = re.compile(r"\b(my (progress|history|stats|log|logs|workout log)|show (me )?(my )?(progress|log|logs|history)|how (many|much) (workouts|sessions) have i)\b")
DID_YOU_MEAN_LABELS = {
    "exercise_technique": "Exercise technique", "beginner_workout": "Make me a workout plan",
    "muscle_gain": "How do I build muscle?", "fat_loss": "How do I lose fat?", "recovery_rest": "How important is rest?",
    "nutrition_general": "What should I eat?", "hydration": "How much water should I drink?",
    "warmup_cooldown": "How should I warm up?", "motivation": "Motivate me", "bot_capabilities": "What can you do?",
}
DEFAULT_SUGGESTIONS = ["How do I do a squat?", "Make me a workout plan", "How do I build muscle?", "What should I eat?"]

# ---------------------------------------------------------- workout flow parsing
WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "once": 1, "twice": 2}
EXPERIENCE_RE = {
    "beginner": re.compile(r"\b(beginner|begginer|new|newbie|novice|never|just start|first time|starting|start out|no experience|not much|few (weeks|months)|0|zero)\b"),
    "intermediate": re.compile(r"\b(intermediate|experienced|advanced|regular|some experience|(\d+|a|one|two|few|couple of) years?|year|pro)\b"),
}
EQUIPMENT_RE = {
    "gym": re.compile(r"\b(gym|machines?|barbells?|full equipment|everything|fitness center|cables?)\b"),
    "dumbbells": re.compile(r"\b(dumbbells?|dumbells?|db|kettlebells?|weights at home|free weights|home weights)\b"),
    "none": re.compile(r"\b(none|no equipment|nothing|bodyweight|body weight|no weights|home|calisthenics|without equipment|no gym)\b"),
}
EQUIPMENT_LABEL = {"none": "no equipment (bodyweight)", "dumbbells": "dumbbells", "gym": "full gym"}
DAY_NAMES_BY_COUNT = {
    2: ["Monday", "Thursday"],
    3: ["Monday", "Wednesday", "Friday"],
    4: ["Monday", "Tuesday", "Thursday", "Friday"],
    5: ["Monday", "Tuesday", "Wednesday", "Friday", "Saturday"],
}
QUICK = {
    "experience": ["Beginner", "Intermediate"],
    "equipment": ["No equipment", "Dumbbells", "Full gym"],
    "days": ["2 days", "3 days", "4 days", "5 days"],
}
QUESTIONS = {
    "experience": "**1/3: What's your training experience?** Are you a *beginner* (less than ~6 months of consistent training) or *intermediate*?",
    "equipment": "**2/3: What equipment do you have?** No equipment (bodyweight at home), dumbbells, or a full gym?",
    "days": "**3/3: How many days per week can you train?** (2–5)",
}
FLOW_STEPS = ["experience", "equipment", "days"]


def _matches_any(patterns, text):
    return any(re.search(p, text) for p in patterns)


def parse_experience(text):
    t = normalize(text)
    if EXPERIENCE_RE["intermediate"].search(t) and not re.search(r"\bnot (very )?(experienced|advanced)\b", t):
        return "intermediate"
    if EXPERIENCE_RE["beginner"].search(t):
        return "beginner"
    return None


def parse_equipment(text):
    t = normalize(text)
    for key in ("gym", "dumbbells", "none"):
        if EQUIPMENT_RE[key].search(t):
            if key == "gym" and re.search(r"\bno gym\b", t):
                continue
            return key
    return None


def parse_days(text):
    t = normalize(text)
    m = re.search(r"\b([1-7])\b", t)
    if m:
        return int(m.group(1))
    for word, n in WORD_NUMBERS.items():
        if re.search(rf"\b{word}\b", t):
            return n
    if re.search(r"\b(every ?day|daily)\b", t):
        return 7
    return None


PARSERS = {"experience": parse_experience, "equipment": parse_equipment, "days": parse_days}


class GymBot:
    def __init__(self, model_path=config.MODEL_PATH, intents_path=config.INTENTS_PATH,
                 exercises_path=config.EXERCISES_PATH, workouts_path=config.WORKOUTS_PATH,
                 threshold=config.CONFIDENCE_THRESHOLD, storage=None):
        try:
            self.model = joblib.load(model_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model not found at {model_path}. Run `python train_model.py` first.") from e
        with open(intents_path, encoding="utf-8") as f:
            self.responses = {i["tag"]: i["responses"] for i in json.load(f)["intents"]}
        with open(exercises_path, encoding="utf-8") as f:
            self.exercises = json.load(f)["exercises"]
        with open(workouts_path, encoding="utf-8") as f:
            workouts = json.load(f)
        self.templates = workouts["templates"]
        self.general_notes = workouts["general_notes"]
        self.threshold = threshold
        self.storage = storage
        # Longest synonyms first so "goblet squat" wins over "squat".
        pairs = [(normalize(s), ex) for ex in self.exercises for s in ex["synonyms"] + [ex["name"]]]
        self._synonyms = sorted(pairs, key=lambda p: len(p[0]), reverse=True)

    # ------------------------------------------------------------ prediction
    def predict_intent(self, text):
        """Return (intent, confidence, top3) where top3 is [(intent, prob), ...]."""
        probs = self.model.predict_proba([text])[0]
        ranked = sorted(zip(self.model.classes_, probs), key=lambda p: p[1], reverse=True)
        return ranked[0][0], float(ranked[0][1]), [(c, round(float(p), 3)) for c, p in ranked[:3]]

    def find_exercise(self, text):
        t = normalize(text)
        for syn, ex in self._synonyms:
            if re.search(rf"\b{re.escape(syn)}\b", t):
                return ex
        return None

    @staticmethod
    def check_safety(text):
        t = text.lower()
        if _matches_any(EMERGENCY_PATTERNS, t):
            return "emergency", EMERGENCY_REPLY
        if _matches_any(UNSAFE_PATTERNS, t):
            return "unsafe_request", UNSAFE_REPLY
        preventive = re.search(r"\b(without|avoid|avoiding|prevent|preventing|reduce)\b.{0,25}\b(pain|injur)", t)
        if _matches_any(INJURY_PATTERNS, t) and not preventive:
            return "medical_injury", INJURY_REPLY
        return None

    # ------------------------------------------------------------- formatting
    @staticmethod
    def format_exercise(ex):
        steps = "\n".join(f"{i}. {s}" for i, s in enumerate(ex["steps"], 1))
        mistakes = "\n".join(f"• {m}" for m in ex["mistakes"])
        return (f"**{ex['name']}**\n"
                f"*Muscles worked:* {', '.join(ex['muscles'])}\n"
                f"*Equipment:* {ex['equipment']} · *Level:* {ex['level']}\n\n"
                f"**How to do it:**\n{steps}\n\n"
                f"**Common mistakes:**\n{mistakes}\n\n"
                f"{{icon:shield}} **Safety:** {ex['safety']}")

    def select_template(self, experience, equipment, days):
        if days <= 3:
            split = "full_body"
        elif days == 5 and experience == "intermediate" and equipment == "gym":
            split = "ppl"
        else:
            split = "upper_lower"
        for t in self.templates:
            if t["equipment"] == equipment and t["split"] == split and experience in t["levels"]:
                return t
        return next(t for t in self.templates if t["equipment"] == equipment and t["split"] == "full_body")

    def build_plan(self, experience, equipment, days):
        notes = []
        if days < 2:
            notes.append("I've set this to 2 days. Two full-body sessions a week is a great minimum for progress.")
            days = 2
        if days > 5:
            notes.append("I've capped this at 5 training days. Rest days are when your body recovers and gets stronger.")
            days = 5
        template = self.select_template(experience, equipment, days)
        day_names = DAY_NAMES_BY_COUNT[days]
        sessions = template["sessions"]
        lines = [f"{{icon:dumbbell}} **Your {template['name']}**",
                 f"*{experience.title()} · {EQUIPMENT_LABEL[equipment]} · {days} days/week*", ""]
        extra_day = template["split"] == "upper_lower" and days == 5
        training_days = day_names[:-1] if extra_day else day_names
        for i, day in enumerate(training_days):
            s = sessions[i % len(sessions)]
            lines.append(f"**{day}: {s['focus']}**")
            lines.extend(f"• {e}" for e in s["exercises"])
            lines.append("")
        if extra_day:
            lines += [f"**{day_names[-1]}: Light Cardio & Mobility**",
                      "• 20–30 min brisk walk, cycling or easy jog", "• 10 min stretching / mobility", ""]
            if experience == "beginner":
                notes.append("As a beginner, 4 strength days plus 1 light cardio day is plenty. More isn't always better.")
        if template["split"] == "ppl":
            notes.append("Rotate Push → Pull → Legs continuously, so next week starts where this one ended.")
        lines.append("**Guidelines:**")
        lines.extend(f"• {n}" for n in self.general_notes)
        if notes:
            lines.append("")
            lines.extend(f"{{icon:note}} {n}" for n in notes)
        lines.append("\n*This is a general template, not personalised medical or coaching advice.*")
        return "\n".join(lines), {"experience": experience, "equipment": equipment, "days": days,
                                  "template": template["id"]}

    # ---------------------------------------------------------- main entry
    def respond(self, message, state=None, user_id=None):
        """Generate a reply. Returns (response_dict, new_state)."""
        state = dict(state or {})
        text = (message or "").strip()

        if not text:
            return self._reply("Please type a message so I can help. {icon:smile}", "validation", source="validation"), state
        if len(text) > config.MAX_MESSAGE_LENGTH:
            return self._reply(f"That message is a bit long. Please keep it under {config.MAX_MESSAGE_LENGTH} characters.",
                               "validation", source="validation"), state
        norm = normalize(text)

        # 1. Safety rules always win and end any active flow.
        safety = self.check_safety(text)
        if safety:
            state.pop("flow", None)
            state.pop("awaiting_exercise", None)
            return self._reply(safety[1], safety[0], confidence=1.0, source="safety_rule"), state

        # 2. Privacy / storage commands.
        if DELETE_DATA_RE.search(norm):
            if self.storage and user_id:
                self.storage.delete_user_data(user_id)
            state = {}
            return self._reply("{icon:trash} Done. I've deleted your saved preferences and workout logs.", "delete_data",
                               source="rule"), state
        if SAVE_PREFS_RE.search(norm):
            return self._save_preferences(state, user_id), state
        if PROGRESS_RE.search(norm):
            return self._progress(user_id), state

        # 3. Workout-plan flow (restart on request, otherwise continue if active).
        if RESTART_RE.search(norm):
            state.update({"flow": "workout", "answers": {}})
            return self._next_question(state, prefix="Sure, let's build a new plan! {icon:zap}")
        if state.get("flow") == "workout":
            return self._continue_flow(text, norm, state)

        # 4. Follow-up answer to "which exercise?"
        exercise = self.find_exercise(text)
        if state.pop("awaiting_exercise", False) and exercise:
            return self._exercise_reply(exercise, 1.0, "entity_match"), state

        # 5. Machine-learning intent classification.
        intent, conf, top3 = self.predict_intent(text)

        if exercise and (intent == "exercise_technique" or conf < self.threshold or len(norm.split()) <= 3):
            return self._exercise_reply(exercise, conf, "model+entity", top3), state

        if conf < self.threshold:
            suggestions = [DID_YOU_MEAN_LABELS[c] for c, _ in top3 if c in DID_YOU_MEAN_LABELS][:3] or DEFAULT_SUGGESTIONS
            return self._reply("{icon:help} I'm not sure I understood that. Could you rephrase it? "
                               "I can help with exercise technique, workout plans, muscle gain, fat loss, recovery, "
                               "warm-ups, hydration and general nutrition.",
                               "fallback", conf, top3, suggestions, source="low_confidence"), state

        if intent == "exercise_technique":
            state["awaiting_exercise"] = True
            names = ["Push-up", "Squat", "Plank", "Lunge", "Deadlift", "Bench press", "Pull-up", "Row"]
            return self._reply(random.choice(self.responses[intent]), intent, conf, top3, names), state

        if intent == "beginner_workout":
            return self._start_flow(text, conf, top3, state, user_id)

        if intent == "medical_injury":
            return self._reply(INJURY_REPLY, intent, conf, top3, source="model"), state

        quick = {
            "greeting": DEFAULT_SUGGESTIONS, "bot_capabilities": DEFAULT_SUGGESTIONS,
            "out_of_scope": DEFAULT_SUGGESTIONS[:3], "thanks": ["Make me a workout plan", "How should I warm up?"],
            "muscle_gain": ["Make me a workout plan", "How much protein do I need?"],
            "fat_loss": ["Make me a workout plan", "Is cardio good for weight loss?"],
            "nutrition_general": ["How much water should I drink?", "How do I build muscle?"],
            "motivation": ["Make me a workout plan", "Show my progress"],
        }.get(intent, [])
        return self._reply(random.choice(self.responses[intent]), intent, conf, top3, quick), state

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _reply(text, intent, confidence=None, top3=None, quick_replies=None, source="model"):
        return {"reply": text, "intent": intent,
                "confidence": round(confidence, 3) if confidence is not None else None,
                "top_intents": top3 or [], "quick_replies": quick_replies or [], "source": source}

    def _exercise_reply(self, ex, conf, source, top3=None):
        others = [e["name"] for e in self.exercises if e["id"] != ex["id"]][:3]
        return self._reply(self.format_exercise(ex), "exercise_technique", conf, top3,
                           [f"How do I do a {n.lower()}?" for n in others[:2]] + ["Make me a workout plan"], source)

    def _start_flow(self, text, conf, top3, state, user_id):
        answers = {step: PARSERS[step](text) for step in FLOW_STEPS}
        answers = {k: v for k, v in answers.items() if v is not None}
        intro = "Let's build your plan! {icon:zap}"
        if not answers and self.storage and user_id:
            saved = self.storage.get_preferences(user_id)
            if saved:
                plan, info = self.build_plan(saved["experience"], saved["equipment"], saved["days"])
                state["last_plan"] = info
                return self._reply("Using your saved preferences:\n\n" + plan, "beginner_workout", conf, top3,
                                   ["Make a different plan"], "flow"), state
        state["flow"] = "workout"
        state["answers"] = answers
        return self._next_question(state, conf, top3, intro)

    def _next_question(self, state, conf=None, top3=None, prefix=""):
        answers = state["answers"]
        for step in FLOW_STEPS:
            if step not in answers:
                state["step"] = step
                msg = (prefix + "\n\n" if prefix else "") + QUESTIONS[step] + "\n\n*(Type \"cancel\" to stop.)*"
                return self._reply(msg, "beginner_workout", conf, top3, QUICK[step], "flow"), state
        plan, info = self.build_plan(answers["experience"], answers["equipment"], answers["days"])
        state.pop("flow", None)
        state.pop("step", None)
        state.pop("answers", None)
        state["last_plan"] = info
        quick = ["{icon:save} Save my preferences", "How do I do a goblet squat?", "How should I warm up?"]
        return self._reply(plan, "beginner_workout", conf, top3, quick, "flow"), state

    def _continue_flow(self, text, norm, state):
        if CANCEL_RE.search(norm):
            for k in ("flow", "step", "answers"):
                state.pop(k, None)
            return self._reply("No problem, I've cancelled the plan builder. What else can I help with?",
                               "cancel", source="flow", quick_replies=DEFAULT_SUGGESTIONS), state
        step = state.get("step", "experience")
        value = PARSERS[step](text)
        if value is None:
            # The user may have asked something unrelated. Answer it if we're confident.
            intent, conf, _ = self.predict_intent(text)
            if conf >= 0.6 and intent not in ("beginner_workout",):
                for k in ("flow", "step", "answers"):
                    state.pop(k, None)
                return self.respond(text, state)
            return self._reply("Sorry, I didn't catch that. " + QUESTIONS[step], "beginner_workout",
                               quick_replies=QUICK[step], source="flow"), state
        state["answers"][step] = value
        for other in FLOW_STEPS:  # the answer might include extra details ("3 days with dumbbells")
            if other not in state["answers"]:
                extra = PARSERS[other](text)
                if extra is not None and not (other == "days" and "year" in norm):
                    state["answers"][other] = extra
        return self._next_question(state)

    def _save_preferences(self, state, user_id):
        info = state.get("last_plan")
        if not info:
            return self._reply("There's no plan to save yet. Say **\"make me a workout plan\"** first.",
                               "save_preferences", source="rule", quick_replies=["Make me a workout plan"])
        if not (self.storage and user_id):
            return self._reply("Saving is not available right now.", "save_preferences", source="rule")
        self.storage.save_preferences(user_id, info["experience"], info["equipment"], info["days"])
        return self._reply("{icon:save} Saved! Next time you ask for a workout plan I'll use these preferences. "
                           "Say **\"delete my data\"** anytime to remove them.", "save_preferences", source="rule")

    def _progress(self, user_id):
        if not (self.storage and user_id):
            return self._reply("Progress tracking is not available right now.", "progress", source="rule")
        s = self.storage.summary(user_id)
        if s["sessions"] == 0:
            return self._reply("You haven't logged any workouts yet. Use the **Progress Log** panel to record a session. {icon:book}",
                               "progress", source="rule")
        return self._reply(f"{{icon:trending}} **Your progress**\n• Sessions logged: **{s['sessions']}**\n"
                           f"• Total training time: **{s['minutes']} min**\n• Sessions in the last 7 days: **{s['this_week']}**\n"
                           f"• Last workout: **{s['last_date']}**\n\nKeep it up. Consistency is everything! {{icon:zap}}",
                           "progress", source="rule")


if __name__ == "__main__":
    # Simple command-line chat for quick testing: python chatbot.py
    bot, st = GymBot(), {}
    print("Gym Bot (type 'quit' to exit)")
    while True:
        try:
            msg = input("You: ")
        except (EOFError, KeyboardInterrupt):
            break
        if msg.strip().lower() in ("quit", "exit"):
            break
        res, st = bot.respond(msg, st)
        print(f"Bot [{res['intent']} {res['confidence']}]: {res['reply']}\n")
