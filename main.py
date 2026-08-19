from Audio_Detection import Detect_Speech_And_Process_Audio,SAMPLE_RATE,CHANNELS,CHUNKSIZE
import sounddevice as sd
import numpy as np
from models.stt import Transcribe
from models.enhancer import Enhance_Audio,Initialize_Enhancer
import queue
import threading
from rag.generation import Generate_Reply
from models.tts import Generate_Speech

SILENCE_THRESHOLD = 0.001

silence_start = None
in_silence = False

audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):

    if status:
        print("STATUS:", status)

    # Get microphone audio
    audio = indata[:, 0].copy()

    # Put audio chunk into queue
    audio_queue.put(audio)

def audio_worker():

    Initialize_Enhancer()

    while True:

        audio = audio_queue.get()

        utterance = Detect_Speech_And_Process_Audio(audio)

        if utterance is not None:

            print("✅ Complete audio ready")
            print("=" * 50)
            print("\n")
            print("🎤 Listening.......")
            # Enhance THIS speech chunk
            # enhanced_audio = Enhance_Audio(utterance)
            
            # Transcribe THIS speech chunk
            stt = Transcribe(utterance)
            
            print(f"Whisper: {stt}")
            answer = Generate_Reply(query=stt,top_k=10)
            print("="*50)
            print("Answer: ",answer)

            if answer is not None:
                tts = Generate_Speech(text=answer,voice='af_bella')

                sd.play(data=tts,samplerate=24000)
                sd.wait()

# ==========================================
# BELLA GREETING
# ==========================================

greeting = Generate_Speech(
    text="Hi, I'm Bella from VisaFlow. How may I help you?",
    voice="af_bella"
)

sd.play(greeting,samplerate=24000)
sd.wait()

# ==========================================
# START MICROPHONE
# ==========================================
worker = threading.Thread(
    target=audio_worker,
    daemon=True
)

worker.start()

try:

    with sd.InputStream(samplerate=SAMPLE_RATE,channels=CHANNELS,dtype="float32",blocksize=CHUNKSIZE,callback=audio_callback):
        print("🎤 Listening.......")
        while True:
            sd.sleep(1000)

except KeyboardInterrupt:
    print("\n🛑 Microphone stopped.")