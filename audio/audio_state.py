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

# Interrupt current playback
tts_interrupt_event = threading.Event()

# Whether TTS is currently playing
tts_is_speaking = False

conversation_history = []