from Audio_Detection import (
    Detect_Speech_And_Process_Audio,
    SAMPLE_RATE,
    CHANNELS,
    CHUNKSIZE
)

import sounddevice as sd
import threading
import time

from models.nemotron import NemotronStreamer
from models.tts import Generate_Speech

import rag.remote_retrieval as remote_retrieval

from audio.audio_state import (
    audio_queue,
    tts_queue,
    query_metrics,
    conversation_history
)

from audio.audio_workers import (
    tts_worker,
    playback_worker
)


# =========================================================
# MICROPHONE CALLBACK
# =========================================================

def audio_callback(indata, frames, time, status):

    if status:

        print(
            "STATUS:",
            status,
            flush=True
        )

    audio = indata[:, 0].copy()

    audio_queue.put(audio)


# =========================================================
# AUDIO WORKER
# =========================================================

def audio_worker():

    nemotron = NemotronStreamer()

    nemotron.start()

    print(
        "🎤 VAD + Nemotron ready",
        flush=True
    )

    print("=" * 50)

    # =====================================================
    # CONTINUOUS AUDIO LOOP
    # =====================================================

    while True:

        audio = audio_queue.get()

        utterance = Detect_Speech_And_Process_Audio(
            audio
        )

        # =================================================
        # NEMOTRON RECEIVES AUDIO CONTINUOUSLY
        # =================================================

        nemotron.add_audio(audio)

        if utterance is None:
            continue

        # =================================================
        # UTTERANCE FINISHED
        # =================================================

        print(
            "\n🛑 VAD detected end of utterance",
            flush=True
        )

        final_query = (
            nemotron.finish_utterance()
        )

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

        # =================================================
        # NO TRANSCRIPT
        # =================================================

        if not final_query:

            print(
                "⚠️ No final transcript received",
                flush=True
            )

            nemotron.reset_transcript()

            print(
                "\n🎤 Listening.......",
                flush=True
            )

            continue

        # =================================================
        # FINAL QUERY
        # =================================================

        print(
            "\n🎯 FINAL QUERY:",
            final_query,
            flush=True
        )

        print("=" * 50)

        # =================================================
        # REMOTE PIPELINE
        #
        # VS CODE
        #    ↓
        # COLAB RETRIEVAL
        #    ↓
        # COLAB RERANKING
        #    ↓
        # KAGGLE QWEN
        #    ↓
        # SENTENCE 1 → TTS
        # SENTENCE 2 → TTS
        # SENTENCE 3 → TTS
        #
        # Retrieve_Remote() handles the whole thing.
        # =================================================

        remote_start = time.perf_counter()

        try:

            result = remote_retrieval.Retrieve_Remote(
                query=final_query,
                conversation_history=conversation_history
            )

        except Exception as e:

            print(
                f"❌ Remote pipeline failed: "
                f"{type(e).__name__}: {e}",
                flush=True
            )

            result = None

        remote_total = (
            time.perf_counter()
            - remote_start
        )

        # =================================================
        # REMOTE PIPELINE FAILED
        # =================================================

        if result is None:

            print(
                "⚠️ Remote retrieval/Qwen "
                "unavailable at the moment.",
                flush=True
            )

            nemotron.reset_transcript()

            print(
                "\n🎤 Listening.......",
                flush=True
            )

            continue

        # =================================================
        # EXTRACT METRICS
        # =================================================

        retrieval_time = result.get(
            "retrieval_time"
        )

        reranking_time = result.get(
            "reranking_time"
        )

        rag_total = result.get(
            "rag_total"
        )

        qwen_total = result.get(
            "qwen_total"
        )

        # =================================================
        # STORE METRICS
        # =================================================

        if retrieval_time is not None:

            query_metrics[
                "retrieval_time"
            ] = retrieval_time

        if qwen_total is not None:

            query_metrics[
                "llm_time"
            ] = qwen_total

        # =================================================
        # PRINT METRICS
        # =================================================

        print(
            "\n" + "=" * 50,
            flush=True
        )

        print(
            f"🌐 Remote pipeline: "
            f"{remote_total:.3f}s",
            flush=True
        )

        if retrieval_time is not None:

            print(
                f"📚 Retrieval: "
                f"{retrieval_time:.3f}s",
                flush=True
            )

        if reranking_time is not None:

            print(
                f"🔄 Reranking: "
                f"{reranking_time:.3f}s",
                flush=True
            )

        if rag_total is not None:

            print(
                f"⚡ Colab RAG total: "
                f"{rag_total:.3f}s",
                flush=True
            )

        if qwen_total is not None:

            print(
                f"🤖 Qwen generation: "
                f"{qwen_total:.3f}s",
                flush=True
            )

        print(
            "=" * 50,
            flush=True
        )

        # =================================================
        # COMPLETE ANSWER
        # =================================================

        answer = result.get(
            "answer",
            ""
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
        # FINAL INFORMATION
        # =================================================

        print(
            "\n📝 Complete Qwen response received.",
            flush=True
        )

        print(
            f"📝 Characters: "
            f"{len(answer)}",
            flush=True
        )

        print(
            "🔊 Sentences were streamed "
            "directly to TTS.",
            flush=True
        )

        # =================================================
        # RESET NEMOTRON
        # =================================================

        nemotron.reset_transcript()

        print(
            "\n🎤 Listening.......",
            flush=True
        )


# =========================================================
# BELLA GREETING
# =========================================================

greeting = Generate_Speech(

    text=(
        " Hi, I'm michael from VisaFlow. "
        "How may I help you?"
    ),

    voice="am_michael"
)

sd.play(
    greeting,
    samplerate=24000
)

sd.wait()


# =========================================================
# START PERSISTENT REMOTE RETRIEVAL WEBSOCKET
# =========================================================

# print(
#     "🔌 Starting persistent Remote Retrieval WebSocket...",
#     flush=True
# )

# remote_retrieval.retrieval_client = (
#     remote_retrieval.RemoteRetrievalWebSocketClient()
# )

# print(
#     "🟢 Persistent Remote Retrieval WebSocket ready.",
#     flush=True
# )


# =========================================================
# START TTS WORKER
# =========================================================

tts_thread = threading.Thread(

    target=tts_worker,

    daemon=True
)

tts_thread.start()


# =========================================================
# START PLAYBACK WORKER
# =========================================================

playback_thread = threading.Thread(

    target=playback_worker,

    daemon=True
)

playback_thread.start()


# =========================================================
# START AUDIO WORKER
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

    # =====================================================
    # CLOSE REMOTE RETRIEVAL WEBSOCKET
    # =====================================================

    if (
        hasattr(
            remote_retrieval,
            "retrieval_client"
        )
        and
        remote_retrieval.retrieval_client
        is not None
    ):

        remote_retrieval.retrieval_client.close()

    print(
        "👋 VoiceRAG stopped.",
        flush=True
    )