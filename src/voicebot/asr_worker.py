"""Private JSON-lines worker. Loading/inference occurs only after a request."""

from __future__ import annotations

import json
import logging
import sys
from contextlib import redirect_stdout

MAX_RESPONSE_BYTES = 1024 * 1024
MAX_SEGMENTS = 2000
MAX_WORDS = 15000


def main() -> None:
    logging.basicConfig(level=logging.ERROR, stream=sys.stderr)
    model = None
    model_key = None
    while True:
        line = sys.stdin.buffer.readline(MAX_RESPONSE_BYTES + 1)
        if not line:
            break
        if len(line) > MAX_RESPONSE_BYTES or not line.endswith(b"\n"):
            _send({"ok": False})
            break
        try:
            request = json.loads(line)
            settings = request["settings"]
            key = (settings["model"], settings["cpu_threads"], settings["cache_dir"])
            # Protect the protocol from libraries which print progress to stdout.
            with redirect_stdout(sys.stderr):
                if model is None or model_key != key:
                    from faster_whisper import WhisperModel

                    model = WhisperModel(
                        settings["model"],
                        device="cpu",
                        compute_type="int8",
                        cpu_threads=settings["cpu_threads"],
                        num_workers=1,
                        download_root=settings["cache_dir"],
                    )
                    model_key = key
                result, _ = model.transcribe(
                    request["path"],
                    task="transcribe",
                    language=settings["language"],
                    multilingual=settings["language"] is None,
                    beam_size=5,
                    vad_filter=True,
                    condition_on_previous_text=False,
                    word_timestamps=True,
                    temperature=(0.0, 0.2, 0.4),
                    hallucination_silence_threshold=2.0,
                    hotwords=settings.get("glossary") or None,
                )
                segments = []
                word_count = 0
                for segment in result:
                    words = [
                        {
                            "start": word.start,
                            "end": word.end,
                            "word": word.word,
                            "probability": word.probability,
                        }
                        for word in (segment.words or [])
                    ]
                    word_count += len(words)
                    if len(segments) >= MAX_SEGMENTS or word_count > MAX_WORDS:
                        raise ValueError("Output limit exceeded")
                    segments.append(
                        {
                            "start": segment.start,
                            "end": segment.end,
                            "text": segment.text,
                            "avg_logprob": segment.avg_logprob,
                            "words": words,
                        }
                    )
            _send({"ok": True, "segments": segments})
        except Exception:
            # Exceptions can contain paths or transcript data; never stringify them.
            sys.stderr.write("ASR worker could not process a request.\n")
            sys.stderr.flush()
            _send({"ok": False})


def _send(response: dict) -> None:
    encoded = (json.dumps(response, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    if len(encoded) > MAX_RESPONSE_BYTES:
        encoded = b'{"ok":false}\n'
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
