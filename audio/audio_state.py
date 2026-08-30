import queue
import threading


# Microphone audio
audio_queue = queue.Queue()

# LLM text → TTS
tts_queue = queue.Queue()

# TTS audio → speaker
tts_audio_queue = queue.Queue()

# Current query metrics
query_metrics = {}

# =========================================================
# INTERRUPTION
# =========================================================

# Set when user starts speaking while TTS is playing.
tts_interrupt_event = threading.Event()

# Set while audio is actually being played.
tts_speaking_event = threading.Event()

# Every response gets a generation ID.
# When interrupted, old audio becomes invalid.
tts_generation_id = 0

conversation_history = []