import queue


# Microphone audio
audio_queue = queue.Queue()

# LLM text → TTS
tts_queue = queue.Queue()

# TTS audio → speaker
tts_audio_queue = queue.Queue()

# Current query metrics
query_metrics = {}

conversation_history = []