from Audio_Detection import (Detect_Speech_And_Process_Audio,SAMPLE_RATE,CHANNELS,CHUNKSIZE)
import sounddevice as sd
import threading
from models.nemotron import NemotronStreamer
from models.tts import Generate_Speech
import rag.remote_retrieval as remote_retrieval
from models.llm import Stream_LLM_To_TTS
import time
from audio.audio_state import audio_queue,tts_queue,query_metrics,conversation_history
from audio.audio_workers import tts_worker,playback_worker
import models.llm as llm
from models.qwen_websocket import QwenWebSocketClient

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

    print("🎤 VAD + Nemotron ready", flush=True)
    print("=" * 50)

    # -----------------------------------------------------
    # Continuous audio loop
    # -----------------------------------------------------

    while True:

        audio = audio_queue.get()

        utterance = Detect_Speech_And_Process_Audio(audio)

        # =================================================
        # Nemotron receives audio continuously
        # =================================================

        nemotron.add_audio(audio)

        if utterance is not None:

            print(
                "\n🛑 VAD detected end of utterance",
                flush=True
            )

            final_query = nemotron.finish_utterance()

            # =================================================
            # QUERY METRICS
            # =================================================

            query_metrics.clear()
            query_metrics.update({
                "query": final_query,
                "query_start": time.perf_counter(),

                "retrieval_time": 0.0,
                "llm_time": 0.0,

                "first_audio_started": None,
                "last_audio_finished": None,

                "tts_generation_total": 0.0,
                "audio_duration": 0.0,
            })

            if final_query:

                print(
                    "\n🎯 FINAL QUERY:",
                    final_query,
                    flush=True
                )

                print("=" * 50)

                # =================================================
                # QUERY → RAG
                # =================================================

                retrieval_start = time.perf_counter()

                rag_response = remote_retrieval.Retrieve_Remote(final_query)
                print("*"*50)
                print(f"Recieved Chunk: {rag_response}")
                print("*"*50)
                if rag_response is None:
                    top_chunk = None
                else:
                    top_chunk = rag_response["results"]
                retrieval_time = (
                    time.perf_counter()
                    - retrieval_start
                )

                query_metrics["retrieval_time"] = retrieval_time

                print(
                    f"⚡ Remote retrieval + reranking: "
                    f"{retrieval_time:.3f} seconds",
                    flush=True
                )

                # =================================================
                # REMOTE RETRIEVAL FAILED
                # =================================================

                if top_chunk is None:

                    print(
                        "⚠️ Remote retrieval unavailable "
                        "at the moment.",
                        flush=True
                    )

                    # No TTS response is coming.
                    # Therefore there is no playback marker
                    # to wait for.

                else:

                    # =================================================
                    # LLM → STREAMING TTS
                    # =================================================
                    llm_start = time.perf_counter()

                    print(
                        f"🟣 ABOUT TO CALL QWEN: "
                        f"{llm_start:.6f}",
                        flush=True
                    )

                    answer = Stream_LLM_To_TTS(
                        query=final_query,
                        context=top_chunk,
                        tts_queue=tts_queue,
                        conversation_history=conversation_history
                    )

                    llm_end = time.perf_counter()

                    print(
                        f"🟣 QWEN FUNCTION RETURNED: "
                        f"{llm_end:.6f}",
                        flush=True
                    )

                    print(
                        f"🟣 MAIN → Stream_LLM_To_TTS: "
                        f"{llm_end - llm_start:.3f}s",
                        flush=True
                    )

                    # =================================================
                    # CONVERSATION HISTORY
                    # =================================================

                    conversation_history.append({
                        "role": "user",
                        "content": final_query
                    })

                    conversation_history.append({
                        "role": "assistant",
                        "content": answer
                    })

                    # =================================================
                    # LLM METRIC
                    # =================================================

                    llm_time = (
                        time.perf_counter()
                        - llm_start
                    )

                    query_metrics["llm_time"] = llm_time

                    print(
                        f"\n🤖 LLM generation: "
                        f"{llm_time:.3f} seconds",
                        flush=True
                    )

            else:

                print(
                    "⚠️ No final transcript received",
                    flush=True
                )

            # ------------------------------------------------
            # Reset Nemotron for next query
            # ------------------------------------------------

            nemotron.reset_transcript()

            print(
                "\n🎤 Listening.......",
                flush=True
            )
# ==========================================
# BELLA GREETING
# ==========================================

greeting = Generate_Speech(
    text=" Hi, I'm Sasha from VisaFlow. How may I help you?",
    voice="af_sky"
)

sd.play(greeting,samplerate=24000)
sd.wait()

# ==========================================
# START PERSISTENT QWEN WEBSOCKET
# ==========================================

print(
    "🔌 Starting persistent Qwen WebSocket...",
    flush=True
)

llm.qwen_client = QwenWebSocketClient()

print(
    "🟢 Persistent Qwen WebSocket ready.",
    flush=True
)

# =========================================================
# START PERSISTENT REMOTE RETRIEVAL WEBSOCKET
# =========================================================

print(
    "🔌 Starting persistent Remote Retrieval WebSocket...",
    flush=True
)

remote_retrieval.retrieval_client = (
    remote_retrieval.RemoteRetrievalWebSocketClient()
)

print(
    "🟢 Persistent Remote Retrieval WebSocket ready.",
    flush=True
)

# ==========================================
# START TTS WORKER
# ==========================================

tts_thread = threading.Thread(
    target=tts_worker,
    daemon=True
)

tts_thread.start()


playback_thread = threading.Thread(
    target=playback_worker,
    daemon=True
)

playback_thread.start()

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

    # ---------------------------------------------
    # CLOSE QWEN
    # ---------------------------------------------

    if llm.qwen_client is not None:

        llm.qwen_client.close()

    # ---------------------------------------------
    # CLOSE REMOTE RAG
    # ---------------------------------------------

    if remote_retrieval.retrieval_client is not None:

        remote_retrieval.retrieval_client.close()

    print(
        "👋 VoiceRAG stopped.",
        flush=True
    )