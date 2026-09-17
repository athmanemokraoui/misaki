"""
misaki/kab.py — Kabyle (Taqbaylit) G2P module for misaki / Kokoro.

"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

try:
    # Preferred: reuse the real shared MToken.
    from .en import MToken  # type: ignore
except Exception:
    @dataclass
    class MToken:
        text: str
        tag: str = ""
        whitespace: str = " "
        phonemes: str | None = None
        underscore: dict = field(default_factory=lambda: {
            "is_head": True,
            "alias": None,
            "stress": None,
            "currency": None,
            "num_flags": "",
            "prespace": False,
            "rating": None,
        })


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

# Kabyle Latin orthography adds diacritics (ɣ ɛ ẓ ṭ ṣ ṛ ḍ č ǧ) to the Latin
# alphabet, and hyphens mark cliticised pronouns (yenna-yas, d-yusa), so
# hyphen-internal words are kept whole; splitting happens on whitespace
# and outer punctuation only.
_WORD_RE = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*", re.UNICODE)
_GAP_RE = re.compile(r"\s+")


def tokenize(text: str) -> list[MToken]:
    """Split text into MTokens, capturing each token's trailing whitespace."""
    text = unicodedata.normalize("NFC", text)
    tokens: list[MToken] = []
    pos = 0
    for m in _WORD_RE.finditer(text):
        start, end = m.span()
        word = m.group(0)
        # whitespace between this word and the next non-space char
        rest = text[end:]
        gap = _GAP_RE.match(rest)
        whitespace = gap.group(0) if gap else ""
        tokens.append(MToken(text=word, whitespace=whitespace))
        pos = end
    return tokens


# ---------------------------------------------------------------------------
# Dictionary
# ---------------------------------------------------------------------------

@dataclass
class DictEntry:
    ipa: str
    count: int = 1


class KabDict:
    """
    Word -> IPA lookup, loaded from a TSV/CSV of (word, ipa[, count]).

    Point this at your own kabyle-g2p-training-data export, or at a
    word-level gold lexicon if you build/use one. Sentence-level data
    needs to be reduced to per-word entries first — this class does not
    do that alignment for you.
    """

    def __init__(self, path: str | Path | None = None):
        self._entries: dict[str, DictEntry] = {}
        if path is not None:
            self.load(path)

    def load(self, path: str | Path, sep: str = "\t") -> None:
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            header = f.readline().strip().split(sep)
            col = {name: i for i, name in enumerate(header)}
            if "word" not in col or "ipa" not in col:
                raise ValueError(
                    f"{path}: expected columns 'word' and 'ipa', got {header}"
                )
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split(sep)
                word = unicodedata.normalize("NFC", parts[col["word"]]).lower()
                ipa = parts[col["ipa"]]
                count = int(parts[col["count"]]) if "count" in col else 1
                existing = self._entries.get(word)
                if existing is None or count > existing.count:
                    self._entries[word] = DictEntry(ipa=ipa, count=count)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, word: str) -> bool:
        return unicodedata.normalize("NFC", word).lower() in self._entries

    def get(self, word: str) -> str | None:
        entry = self._entries.get(unicodedata.normalize("NFC", word).lower())
        return entry.ipa if entry else None


# ---------------------------------------------------------------------------
# Fallback for out-of-dictionary words
# ---------------------------------------------------------------------------

UNRESOLVED = "\u2047"  # "⁇" — deliberately loud, never mistaken for real IPA


def _fallback_g2p(word: str) -> str:
    """
    Placeholder. Replace with either:
      (a) cited orthography->IPA rules for Kabyle, or
      (b) a seq2seq model trained on your dictionary data — misaki's own
          TODOs list exactly this pattern for other languages.
    Returns UNRESOLVED for now so gaps are loud, not silently wrong.
    """
    return UNRESOLVED


# ---------------------------------------------------------------------------
# Public G2P interface
# ---------------------------------------------------------------------------

class G2P:
    def __init__(self, dict_path: str | Path | None = None, kab_dict: KabDict | None = None):
        self.dict = kab_dict if kab_dict is not None else KabDict(dict_path)

    def __call__(self, text: str) -> tuple[str, list[MToken]]:
        tokens = tokenize(text)
        phoneme_parts = []
        for tok in tokens:
            ipa = self.dict.get(tok.text)
            if ipa is None:
                ipa = _fallback_g2p(tok.text)
            tok.phonemes = ipa
            phoneme_parts.append(ipa + tok.whitespace)
        phonemes = "".join(phoneme_parts).rstrip()
        return phonemes, tokens


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("usage: python kab.py <dict.tsv> <text>")
        raise SystemExit(1)

    g2p = G2P(dict_path=sys.argv[1])
    phonemes, tokens = g2p(sys.argv[2])
    print(phonemes)
    for t in tokens:
        print(t.text, "->", t.phonemes)
