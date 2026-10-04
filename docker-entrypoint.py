"""Create writable model cache on Railway's root-owned mounted volume; then drop privileges."""

import os
from pathlib import Path

cache = Path(os.environ.get("MODEL_CACHE_DIR", "/data/models"))
cache.mkdir(parents=True, exist_ok=True)
if os.getuid() == 0:
    os.chown(cache, 10001, 10001)
    os.setgroups([])
    os.setgid(10001)
    os.setuid(10001)
os.execvp("python", ["python", "-m", "voicebot"])
