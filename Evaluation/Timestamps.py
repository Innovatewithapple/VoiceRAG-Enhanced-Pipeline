import time
import json
from datetime import datetime

LOG_PATH = "voice_rag_latency_log.jsonl"

def log_query(query, retrieval_time, llm_time, tts_time, processing_total, ttfa=None, playback_duration=None, notes=""):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "retrieval_time": round(retrieval_time, 3),
        "llm_time": round(llm_time, 3),
        "tts_time": round(tts_time, 3),
        "processing_total": round(processing_total, 3),   # the real 6.4s-style number
        "ttfa": round(ttfa, 3) if ttfa else None,
        "playback_duration": round(playback_duration, 3) if playback_duration else None,
        "notes": notes,   # e.g. "before streaming TTS" / "after switching to X model"
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry



# import pandas as pd

# df = pd.read_json(LOG_PATH, lines=True)

# # Compare before/after an optimization using the notes field
# summary = df.groupby("notes")[["retrieval_time", "llm_time", "tts_time", "processing_total"]].agg(["mean", "median"])
# print(summary)

# This gives you clean, aggregated numbers (not just one lucky/unlucky single run) whenever you make a change — e.g., tag runs as "baseline" before an optimization and "streaming_tts" after, then compare the two groups' averages directly.

# For GitHub specifically

# Once you've got enough logged runs, a simple matplotlib bar chart or table in your README — "baseline vs. optimized" per stage — is exactly the kind of visual evidence that makes a project stand out. Something like:

# Stage	Before	After	Improvement
# Retrieval	0.91s	...	...
# LLM	1.47s	...	...
# TTS	4.05s	...	...
# Total processing	6.43s	...	...