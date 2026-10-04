from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class Transcript:
    text: str
    uncertain_ranges: tuple[str, ...] = ()


def timestamp(seconds: float) -> str:
    minutes, seconds = divmod(max(0, int(seconds)), 60)
    return f"{minutes:02d}:{seconds:02d}"


def build_transcript(segments: list[dict]) -> Transcript:
    pieces: list[str] = []
    uncertain: list[str] = []
    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        # Scores are heuristics, never an assertion that the word is correct/incorrect.
        words = segment.get("words") or []
        low_words = [word for word in words if word.get("probability", 1) < 0.35]
        unsure = segment.get("avg_logprob", 0) < -0.85 or bool(low_words)
        span = f"{timestamp(segment['start'])}–{timestamp(segment['end'])}"
        if unsure:
            pieces.append(f"[неуверенно {span}: {text}]")
            uncertain.append(span)
        else:
            pieces.append(text)
    return Transcript(" ".join(pieces), tuple(uncertain))


def split_text(text: str, limit: int = 3500) -> list[str]:
    """Split with a UTF-16 budget (Telegram entity length), preserving all characters."""
    if limit < 2:
        raise ValueError("limit must be at least 2")
    chunks = []
    start = 0
    while start < len(text):
        end = start
        units = 0
        while end < len(text):
            width = 2 if ord(text[end]) > 0xFFFF else 1
            if units + width > limit:
                break
            units += width
            end += 1
        if end < len(text):
            boundary = max(text.rfind("\n", start, end), text.rfind(" ", start, end))
            if boundary > start + (end - start) // 2:
                end = boundary + 1
        chunks.append(text[start:end])
        start = end
    return chunks


def translation_chunks(text: str) -> list[str]:
    # Short chunks avoid truncating translation on the provider's output budget.
    return split_text(text, 3000)


def translation_input(text: str) -> str:
    return json.dumps({"transcript": text}, ensure_ascii=False)
