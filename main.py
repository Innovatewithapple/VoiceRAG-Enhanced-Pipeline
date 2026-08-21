from Audio_Detection import (Detect_Speech_And_Process_Audio,SAMPLE_RATE,CHANNELS,CHUNKSIZE)
import sounddevice as sd
import queue
import threading
from models.nemotron import NemotronStreamer
# from rag.generation import Generate_Reply
from models.tts import Generate_Speech
import sounddevice as sd
import numpy as np
from rag.remote_retrieval import Retrieve_Remote
from models.llm import Generate_LLM_Response
import time
from Evaluation.Timestamps import log_query


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
            overall_start = time.perf_counter()
            if final_query:
                
                print("\n🎯 FINAL QUERY:",final_query,flush=True)
                print("=" * 50)

                # =================================================
                # QUERY → RAG
                # =================================================
                retrieval_start = time.perf_counter()
                top_chunk = Retrieve_Remote(final_query)
                retrieval_time = (time.perf_counter() - retrieval_start)
                print(f"⚡ Remote retrieval + reranking: {retrieval_time:.3f} seconds",flush=True)

                if top_chunk is None:
                    print("⚠️ Remote retrieval unavailable at the moment.",flush=True)
                    
                else:
                    llm_start = time.perf_counter()
                    answer = Generate_LLM_Response(query=final_query,context=top_chunk)
                    llm_time = (time.perf_counter() - llm_start)
                    print(f"🤖 LLM generation: {llm_time:.3f} seconds",flush=True)
                    # answer = Generate_Reply(query=final_query,top_k=10)

                    print("\n🤖 Answer:",answer,flush=True)
                    if answer is not None:
                        tts_start = time.perf_counter()
                        tts = Generate_Speech(text=answer,voice='am_michael')
                        tts_time = (time.perf_counter() - tts_start)
                        print(f"🔊 TTS generation: {tts_time:.3f} seconds")

                        playback_start = time.perf_counter()
                        sd.play(data=tts,samplerate=24000)
                        sd.wait()
                        playback_duration = (time.perf_counter() - playback_start)
                        print(f"\n📡 playback_duration: {playback_duration:.3f} seconds",flush=True)

            else:
                print("⚠️ No final transcript received",flush=True)

            # ------------------------------------------------
            # Reset Nemotron for next query
            # ------------------------------------------------
            overall_time = (time.perf_counter() - overall_start)
            print(f"\n⏱️ TOTAL QUERY PIPELINE: {overall_time:.3f} seconds",flush=True)
            print("=" * 50)

            # ---- Log everything ----
            log_query(
                query=final_query,
                retrieval_time=retrieval_time,
                llm_time=llm_time,
                tts_time=tts_time,
                processing_total=overall_time,
                playback_duration=playback_duration,
                notes="baseline" 
            )
            nemotron.reset_transcript()

            print("\n🎤 Listening.......",flush=True)

# ==========================================
# BELLA GREETING
# ==========================================

greeting = Generate_Speech(
    text="Hi, I'm Edith from VisaFlow. How may I help you?",
    voice="am_michael"
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