import re
import unicodedata

from .config import LIMIT

LEXICON = (
    "idiot",
    "moron",
    "asshole",
    "stupid",
    "dumbass",
    "fucker",
    "shithead",
    "bastard",
)
LEET = str.maketrans(
    {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"}
)
WORDS = re.compile(r"[^\W_]+", re.UNICODE)


def validate(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter a message with at least one visible character.")
    if len(text) > LIMIT:
        raise ValueError(f"Use {LIMIT:,} characters or fewer.")
    return text.strip()


def clean_text(text):
    # Keep case and punctuation for the pretrained cased WordPiece tokenizer.
    return " ".join(unicodedata.normalize("NFKC", text[:LIMIT]).split())


def tokens(text):
    value = clean_text(text).casefold().replace("i\u0307", "i").replace("ı", "i")

    # Transform candidate words only. Keep standalone numbers and usernames intact.
    def repair(m):
        word = m.group()
        if word.startswith(("@", "$")) or not any(c.isalpha() for c in word):
            return word
        return word.translate(LEET)

    value = re.sub(r"[\w@$]+", repair, value)
    return WORDS.findall(re.sub(r"([^\W\d_])\1{2,}", r"\1\1", value))


def edit_distance(a, b):
    row = list(range(len(b) + 1))
    for i, char in enumerate(a, start=1):
        nxt = [i]
        for j, other in enumerate(b, start=1):
            nxt.append(min(row[j] + 1, nxt[-1] + 1, row[j - 1] + (char != other)))
        row = nxt
    return row[-1]


def lexicon_hit(text):
    return any(
        abs(len(w) - len(key)) <= 1 and edit_distance(w, key) <= 1
        for w in set(tokens(text))
        if len(w) >= 4
        for key in LEXICON
    )


def displayed(text, offensive):
    return "••••  Message hidden" if offensive else text
