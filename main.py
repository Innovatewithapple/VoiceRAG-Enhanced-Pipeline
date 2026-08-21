from Audio_Detection import (Detect_Speech_And_Process_Audio,SAMPLE_RATE,CHANNELS,CHUNKSIZE)
import sounddevice as sd
import queue
import threading
from models.nemotron import NemotronStreamer
from rag.generation import Generate_Reply
from models.tts import Generate_Speech
import sounddevice as sd
import numpy as np


# =========================================================
# MICROPHONE QUEUE
# =========================================================

audio_queue = queue.Queue()

# =========================================================
# MICROPHONE CALLBACK
# =========================================================

def audio_callback(indata,frames,time,status):
    if status:
        print("STATUS:",status,flush=True)

    audio = indata[:, 0].copy()
    audio_queue.put(audio)


def audio_worker():
    nemotron = NemotronStreamer()
    nemotron.start()

    print("🎤 VAD + Nemotron ready",flush=True)
    print("=" * 50)

    # -----------------------------------------------------
    # Continuous audio loop
    # -----------------------------------------------------

    while True:
        audio = audio_queue.get()
        utterance = (Detect_Speech_And_Process_Audio(audio))

        # =================================================
        # Nemotron receives audio continuously
        # =================================================

        nemotron.add_audio(audio) 

        if utterance is not None:

            print("\n🛑 VAD detected end of utterance",flush=True)
            final_query = (nemotron.finish_utterance())

            if final_query:

                print("\n🎯 FINAL QUERY:",final_query,flush=True)

                print("=" * 50)

                # =================================================
                # QUERY → RAG
                # =================================================

                answer = Generate_Reply(query=final_query,top_k=10)

                print("\n🤖 Answer:",answer,flush=True)
                if answer is not None:
                    tts = Generate_Speech(text=answer,voice='af_bella')

                    sd.play(data=tts,samplerate=24000)
                    sd.wait()

            else:
                print("⚠️ No final transcript received",flush=True)

            # ------------------------------------------------
            # Reset Nemotron for next query
            # ------------------------------------------------

            nemotron.reset_transcript()

            print("\n🎤 Listening.......",flush=True)

# ==========================================
# BELLA GREETING
# ==========================================

greeting = Generate_Speech(
    text="Hi, I'm Edith from VisaFlow. How may I help you?",
    voice="af_bella"
)

sd.play(greeting,samplerate=24000)
sd.wait()

# =========================================================
# START WORKER
# =========================================================

worker = threading.Thread(
    target=audio_worker,
    daemon=True
)

worker.start()


# =========================================================
# START MICROPHONE
# =========================================================

try:

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=CHUNKSIZE,
        callback=audio_callback,
    ):

        print(
            "🎤 Listening.......",
            flush=True
        )

        while True:

            sd.sleep(1000)

except KeyboardInterrupt:

    print(
        "\n🛑 Microphone stopped.",
        flush=True
    )