"""Train, evaluate and save the intent classification pipeline.

Steps
  1. Load and validate data/intents.json
  2. Build labelled (sentence, intent) examples and remove duplicates
  3. Stratified train/test split, moving near-duplicates out of the test set
  4. Compare candidate classifiers with k-fold cross-validation
  5. Evaluate the best model on the held-out test set
  6. Save reports (metrics, classification report, confusion matrix)
  7. Refit on all data and save the fitted pipeline with joblib

Run:  python train_model.py
"""
import json
import sys
import time
from collections import Counter

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

import config
from nlp_utils import normalize


# ---------------------------------------------------------------- data loading
def load_intents(path=config.INTENTS_PATH):
    """Load intents.json and validate its structure. Raises ValueError on problems."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    intents = data.get("intents")
    if not isinstance(intents, list) or not intents:
        raise ValueError("intents.json must contain a non-empty 'intents' list")
    seen_tags = set()
    for i, intent in enumerate(intents):
        for key in ("tag", "patterns", "responses"):
            if key not in intent:
                raise ValueError(f"Intent #{i} is missing '{key}'")
        if intent["tag"] in seen_tags:
            raise ValueError(f"Duplicate tag: {intent['tag']}")
        seen_tags.add(intent["tag"])
        if len(intent["patterns"]) < 5:
            raise ValueError(f"Intent '{intent['tag']}' needs at least 5 patterns")
        if not intent["responses"]:
            raise ValueError(f"Intent '{intent['tag']}' has no responses")
    return intents


def build_examples(intents):
    """Return (texts, labels) with exact duplicates (after normalisation) removed."""
    texts, labels, seen = [], [], {}
    conflicts = []
    for intent in intents:
        for pattern in intent["patterns"]:
            key = normalize(pattern)
            if not key:
                continue
            if key in seen:
                if seen[key] != intent["tag"]:
                    conflicts.append((pattern, seen[key], intent["tag"]))
                continue
            seen[key] = intent["tag"]
            texts.append(pattern)
            labels.append(intent["tag"])
    if conflicts:
        print("WARNING: same sentence used for different intents:")
        for c in conflicts:
            print("   ", c)
    return texts, labels


# ------------------------------------------------------------------- pipeline
def build_features():
    """Word n-grams capture meaning; character n-grams make the model robust to typos."""
    return FeatureUnion([
        ("word", TfidfVectorizer(preprocessor=normalize, analyzer="word",
                                 ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
        ("char", TfidfVectorizer(preprocessor=normalize, analyzer="char_wb",
                                 ngram_range=(2, 5), sublinear_tf=True, min_df=1)),
    ])


def candidate_models():
    return {
        "Logistic Regression": LogisticRegression(C=30, max_iter=5000, class_weight="balanced"),
        "Linear SVM (calibrated)": CalibratedClassifierCV(LinearSVC(C=1.0, class_weight="balanced"), cv=3),
        "Complement Naive Bayes": ComplementNB(alpha=0.3),
    }


def build_pipeline(classifier):
    return Pipeline([("features", build_features()), ("clf", classifier)])


# ------------------------------------------------------------ split helpers
def remove_near_duplicates(X_train, X_test, y_train, y_test, threshold):
    """Move test sentences that are almost identical to a training sentence into the
    training set, so the test score is not inflated by near-copies."""
    vec = TfidfVectorizer(preprocessor=normalize).fit(X_train + X_test)
    sims = cosine_similarity(vec.transform(X_test), vec.transform(X_train))
    max_sim = sims.max(axis=1)
    keep = max_sim < threshold
    moved = int((~keep).sum())
    X_train = X_train + [x for x, k in zip(X_test, keep) if not k]
    y_train = y_train + [y for y, k in zip(y_test, keep) if not k]
    X_test = [x for x, k in zip(X_test, keep) if k]
    y_test = [y for y, k in zip(y_test, keep) if k]
    return X_train, X_test, y_train, y_test, moved


def save_confusion_matrix(y_true, y_pred, labels, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted intent")
    ax.set_ylabel("True intent")
    ax.set_title("Intent Classifier: Confusion Matrix (held-out test set)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------- main
def main():
    start = time.time()
    config.MODEL_DIR.mkdir(exist_ok=True)
    config.REPORTS_DIR.mkdir(exist_ok=True)

    print("1) Loading and validating intents ...")
    intents = load_intents()
    texts, labels = build_examples(intents)
    counts = Counter(labels)
    print(f"   {len(intents)} intents, {len(texts)} unique examples")
    for tag, n in sorted(counts.items()):
        print(f"     {tag:<20} {n}")

    print("\n2) Splitting data (stratified, test size = %.0f%%) ..." % (config.TEST_SIZE * 100))
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=config.TEST_SIZE, stratify=labels, random_state=config.RANDOM_STATE)
    X_train, X_test, y_train, y_test, moved = remove_near_duplicates(
        X_train, X_test, y_train, y_test, config.NEAR_DUPLICATE_SIMILARITY)
    print(f"   train = {len(X_train)}, test = {len(X_test)} "
          f"({moved} near-duplicate test sentences moved to train)")

    print(f"\n3) Comparing models with {config.CV_FOLDS}-fold cross-validation (macro F1) ...")
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    cv_results = {}
    for name, clf in candidate_models().items():
        scores = cross_val_score(build_pipeline(clf), X_train, y_train, cv=cv, scoring="f1_macro")
        cv_results[name] = {"mean": float(scores.mean()), "std": float(scores.std())}
        print(f"   {name:<26} {scores.mean():.3f} ± {scores.std():.3f}")
    best_name = max(cv_results, key=lambda n: cv_results[n]["mean"])
    print(f"   -> Selected: {best_name}")

    print("\n4) Evaluating the selected model on the held-out test set ...")
    model = build_pipeline(candidate_models()[best_name]).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    probs = model.predict_proba(X_test).max(axis=1)
    acc = accuracy_score(y_test, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    confident = probs >= config.CONFIDENCE_THRESHOLD
    coverage = float(confident.mean())
    conf_acc = float((np.array(y_pred)[confident] == np.array(y_test)[confident]).mean()) if confident.any() else 0.0

    report = classification_report(y_test, y_pred, zero_division=0)
    print(report)
    print(f"   Accuracy           : {acc:.3f}")
    print(f"   Macro precision    : {p:.3f}")
    print(f"   Macro recall       : {r:.3f}")
    print(f"   Macro F1           : {f1:.3f}")
    print(f"   Confidence >= {config.CONFIDENCE_THRESHOLD}: answers {coverage:.1%} of test questions "
          f"with {conf_acc:.1%} accuracy (rest get a 'please rephrase' fallback)")

    tags = sorted(counts)
    save_confusion_matrix(y_test, y_pred, tags, config.REPORTS_DIR / "confusion_matrix.png")
    errors = [{"text": x, "true": t, "predicted": pr}
              for x, t, pr in zip(X_test, y_test, y_pred) if t != pr]
    metrics = {
        "selected_model": best_name,
        "cross_validation_macro_f1": cv_results,
        "test_set": {
            "size": len(X_test), "accuracy": acc, "macro_precision": p, "macro_recall": r, "macro_f1": f1,
            "confidence_threshold": config.CONFIDENCE_THRESHOLD,
            "coverage_at_threshold": coverage, "accuracy_when_confident": conf_acc,
        },
        "dataset": {"intents": len(intents), "examples": len(texts), "per_intent": dict(counts),
                    "train_size": len(X_train), "near_duplicates_moved": moved},
        "misclassified_examples": errors,
    }
    with open(config.REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(config.REPORTS_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Selected model: {best_name}\n\n{report}\n")
    print(f"\n   Reports saved to {config.REPORTS_DIR}/")

    print("\n5) Refitting on all data and saving the pipeline ...")
    final_model = build_pipeline(candidate_models()[best_name]).fit(texts, labels)
    joblib.dump(final_model, config.MODEL_PATH)
    with open(config.MODEL_META_PATH, "w", encoding="utf-8") as f:
        json.dump({"model": best_name, "classes": list(final_model.classes_),
                   "trained_on_examples": len(texts), "test_accuracy": acc, "test_macro_f1": f1},
                  f, indent=2)
    print(f"   Saved {config.MODEL_PATH.relative_to(config.BASE_DIR)}")
    print(f"\nDone in {time.time() - start:.1f}s")
    return metrics


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)
