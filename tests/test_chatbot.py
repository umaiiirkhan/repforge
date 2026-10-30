"""Tests for the chatbot logic (covers the testing table in the project document)."""
import json

import pytest

import config
from chatbot import GymBot, parse_days, parse_equipment, parse_experience
from nlp_utils import normalize
from train_model import build_examples, load_intents


# --------------------------------------------------------------- data checks
def test_intents_file_is_valid():
    intents = load_intents()
    assert len(intents) >= 10
    for intent in intents:
        assert len(intent["patterns"]) >= 25, intent["tag"]
        assert intent["responses"]


def test_no_sentence_used_for_two_intents():
    intents = load_intents()
    owner = {}
    for intent in intents:
        for p in intent["patterns"]:
            key = normalize(p)
            assert owner.get(key, intent["tag"]) == intent["tag"], f"'{p}' is in two intents"
            owner[key] = intent["tag"]


def test_exercises_file_is_complete():
    data = json.loads(config.EXERCISES_PATH.read_text(encoding="utf-8"))
    for ex in data["exercises"]:
        for key in ("id", "name", "synonyms", "muscles", "steps", "mistakes", "safety"):
            assert ex.get(key), f"{ex.get('id')} missing {key}"


def test_build_examples_removes_duplicates():
    texts, labels = build_examples([{"tag": "a", "patterns": ["Hi!", "hi", "hello"], "responses": ["x"]}])
    assert texts == ["Hi!", "hello"] and labels == ["a", "a"]


def test_normalize():
    assert normalize("How do I do PUSH-UPS??") == "how do i do push up"
    assert normalize("I'm can't") == "i am can not"
    assert normalize("") == ""
    assert normalize(None) == ""


# ------------------------------------------------------- testing-table cases
def test_greeting(bot):
    res, _ = bot.respond("hello")
    assert res["intent"] == "greeting"


def test_exercise_technique_returns_matching_guidance(bot):
    res, _ = bot.respond("How do I perform push-ups?")
    assert res["intent"] == "exercise_technique"
    assert "Push-Up" in res["reply"]
    res, _ = bot.respond("what's the correct squat form")
    assert "Squat" in res["reply"]


def test_longest_exercise_synonym_wins(bot):
    assert bot.find_exercise("how to do a goblet squat")["id"] == "goblet_squat"
    assert bot.find_exercise("romanian deadlift tips")["id"] == "romanian_deadlift"
    assert bot.find_exercise("I want to grow") is None  # "row" must not match inside "grow"


def test_technique_question_without_exercise_asks_which(bot):
    res, state = bot.respond("teach me the proper technique")
    assert state.get("awaiting_exercise")
    res, state = bot.respond("plank", state)
    assert "Plank" in res["reply"]
    assert not state.get("awaiting_exercise")


def test_beginner_workout_flow(bot):
    res, state = bot.respond("Make me a workout plan")
    assert state["flow"] == "workout" and state["step"] == "experience"
    res, state = bot.respond("Beginner", state)
    assert state["step"] == "equipment"
    res, state = bot.respond("No equipment", state)
    assert state["step"] == "days"
    res, state = bot.respond("3 days", state)
    assert "flow" not in state
    assert "Bodyweight Full-Body Plan" in res["reply"]
    assert "Monday" in res["reply"] and "Friday" in res["reply"]


def test_workout_flow_prefills_details_from_first_message(bot):
    res, state = bot.respond("I'm intermediate with a gym, 5 days, give me a plan")
    assert "Push/Pull/Legs" in res["reply"]


def test_workout_flow_cancel(bot):
    _, state = bot.respond("create a workout schedule for me")
    res, state = bot.respond("cancel", state)
    assert res["intent"] == "cancel" and "flow" not in state


def test_workout_flow_reasks_on_unclear_answer(bot):
    _, state = bot.respond("make me a workout plan")
    res, state = bot.respond("hmm", state)
    assert state["flow"] == "workout" and "didn't catch" in res["reply"]


@pytest.mark.parametrize("exp,equip,days,expected", [
    ("beginner", "none", 2, "bodyweight_full_body"),
    ("beginner", "dumbbells", 3, "dumbbell_full_body"),
    ("beginner", "gym", 4, "gym_upper_lower"),
    ("beginner", "gym", 5, "gym_upper_lower"),       # beginners don't get PPL
    ("intermediate", "gym", 5, "gym_push_pull_legs"),
    ("intermediate", "none", 4, "bodyweight_upper_lower"),
])
def test_template_selection(bot, exp, equip, days, expected):
    assert bot.select_template(exp, equip, days)["id"] == expected


def test_plan_caps_days(bot):
    plan, info = bot.build_plan("beginner", "gym", 7)
    assert info["days"] == 5 and "capped" in plan


def test_unrelated_question_uses_fallback(bot):
    res, _ = bot.respond("what's the weather today")
    assert res["intent"] in ("out_of_scope", "fallback")


def test_empty_input_shows_validation(bot):
    for msg in ("", "   ", None):
        res, _ = bot.respond(msg)
        assert res["intent"] == "validation"


def test_too_long_input(bot):
    res, _ = bot.respond("a" * (config.MAX_MESSAGE_LENGTH + 1))
    assert res["intent"] == "validation"


def test_low_confidence_requests_clarification(bot):
    res, _ = bot.respond("qwzx plorb vintak")
    assert res["intent"] == "fallback"
    assert res["confidence"] < config.CONFIDENCE_THRESHOLD
    assert "rephrase" in res["reply"]


def test_repeated_requests_are_stable(bot):
    state = {}
    for _ in range(25):
        res, state = bot.respond("how much protein do i need", state)
        assert res["intent"] == "nutrition_general"


# ------------------------------------------------------------------ safety
@pytest.mark.parametrize("msg", ["I have chest pain when I run", "I fainted at the gym yesterday", "I can't breathe properly"])
def test_emergency_messages(bot, msg):
    res, _ = bot.respond(msg)
    assert res["intent"] == "emergency" and "medical help" in res["reply"]


@pytest.mark.parametrize("msg", ["my knee hurts when I squat", "I sprained my ankle", "I think I tore my shoulder"])
def test_injury_messages(bot, msg):
    res, _ = bot.respond(msg)
    assert res["intent"] == "medical_injury"


@pytest.mark.parametrize("msg", ["how to lose 10 kg in a week", "should I take steroids", "I want to stop eating to lose weight"])
def test_unsafe_requests_are_declined(bot, msg):
    res, _ = bot.respond(msg)
    assert res["intent"] == "unsafe_request"


def test_preventive_pain_question_is_not_flagged(bot):
    res, _ = bot.respond("how do i squat without knee pain")
    assert res["intent"] == "exercise_technique"


def test_safety_interrupts_workout_flow(bot):
    _, state = bot.respond("make me a workout plan")
    res, state = bot.respond("I have chest pain", state)
    assert res["intent"] == "emergency" and "flow" not in state


# ----------------------------------------------------- parsers & storage flow
def test_parsers():
    assert parse_experience("I'm a total beginner") == "beginner"
    assert parse_experience("training for 2 years") == "intermediate"
    assert parse_equipment("just dumbbells at home") == "dumbbells"
    assert parse_equipment("I go to a gym") == "gym"
    assert parse_equipment("no equipment") == "none"
    assert parse_days("4 days") == 4
    assert parse_days("three times a week") == 3
    assert parse_days("whenever") is None


def test_save_preferences_and_reuse(bot, storage):
    _, state = bot.respond("I'm a beginner with dumbbells, 3 days, make me a workout plan", user_id="u1")
    res, state = bot.respond("save my preferences", state, user_id="u1")
    assert res["intent"] == "save_preferences"
    assert storage.get_preferences("u1") == {"experience": "beginner", "equipment": "dumbbells", "days": 3}
    res, _ = bot.respond("make me a workout plan", {}, user_id="u1")
    assert "saved preferences" in res["reply"]


def test_progress_summary_and_delete(bot, storage):
    res, _ = bot.respond("show my progress", user_id="u2")
    assert "haven't logged" in res["reply"]
    storage.add_log("u2", "Full Body A", 45)
    res, _ = bot.respond("show my progress", user_id="u2")
    assert "Sessions logged: **1**" in res["reply"]
    bot.respond("delete my data", user_id="u2")
    assert storage.get_logs("u2") == []


def test_model_meets_minimum_quality():
    metrics = json.loads((config.REPORTS_DIR / "metrics.json").read_text())
    assert metrics["test_set"]["accuracy"] >= 0.75
    assert metrics["test_set"]["accuracy_when_confident"] >= 0.85
