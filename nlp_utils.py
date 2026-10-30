"""Text preprocessing shared by training and prediction.

The same `normalize` function is used inside the saved Scikit-learn pipeline,
so text is cleaned identically at training time and at prediction time.
"""
import re
import unicodedata

# Common spelling variants / slang mapped to one canonical form.
_REPLACEMENTS = {
    r"\bpush[\s-]?ups?\b": "push up",
    r"\bpress[\s-]?ups?\b": "push up",
    r"\bpull[\s-]?ups?\b": "pull up",
    r"\bchin[\s-]?ups?\b": "chin up",
    r"\bsit[\s-]?ups?\b": "sit up",
    r"\bwork[\s-]?outs?\b": "workout",
    r"\bu\b": "you",
    r"\bur\b": "your",
    r"\bpls\b|\bplz\b": "please",
    r"\bwanna\b": "want to",
    r"\bgonna\b": "going to",
    r"\bim\b": "i am",
    r"\bi'm\b": "i am",
    r"\bdont\b": "do not",
    r"\bcant\b": "can not",
    r"\bwhats\b": "what is",
    r"\bkgs?\b": "kg",
}

_CONTRACTIONS = {
    "can't": "can not",
    "won't": "will not",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'ll": " will",
    "'ve": " have",
    "'d": " would",
}


def normalize(text: str) -> str:
    """Lowercase, strip accents/emoji, expand contractions and slang, collapse spaces."""
    if not isinstance(text, str):
        return ""
    text = text.replace("’", "'").replace("‘", "'")
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    for pattern, repl in _REPLACEMENTS.items():
        text = re.sub(pattern, repl, text)
    for contraction, expansion in _CONTRACTIONS.items():
        text = text.replace(contraction, expansion)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
