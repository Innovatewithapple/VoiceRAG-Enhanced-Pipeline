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
from models.llm import Generate_LLM_Response, Stream_LLM_To_TTS
import time
from Evaluation.Timestamps import log_query


# =========================================================
# MICROPHONE QUEUE
# =========================================================

audio_queue = queue.Queue()              # microphone
tts_queue = queue.Queue()                # text → Kokoro
tts_audio_queue = queue.Queue()          # Kokoro → speaker
query_metrics = {}

def get_silence_duration(audio, threshold=0.005):

    audio = np.asarray(audio)

    if audio.ndim > 1:
        audio = audio[:, 0]

    active = np.where(
        np.abs(audio) > threshold
    )[0]

    if len(active) == 0:
        return (
            len(audio) / 24000,
            len(audio) / 24000
        )

    start = active[0]
    end = active[-1]

    leading = start / 24000
    trailing = (len(audio) - end - 1) / 24000

    return leading, trailing


def trim_silence(
    audio,
    threshold=0.005,
    keep_trailing=0.38
):

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    if audio.ndim > 1:
        audio = audio[:, 0]

    active = np.where(
        np.abs(audio) > threshold
    )[0]

    if len(active) == 0:
        return None

    start = active[0]
    end = active[-1] + 1

    # Keep a small amount of trailing silence
    keep_samples = int(
        keep_trailing * 24000
    )

    end = min(
        end + keep_samples,
        len(audio)
    )

    return audio[start:end]

def tts_worker():

    while True:

        sentence = tts_queue.get()

        if sentence is None:
            # Tell playback that this query is finished
            tts_audio_queue.put(None)
            tts_queue.task_done()
            break

        print(
            f"\n🔊 Speaking: {sentence}",
            flush=True
        )

        tts_start = time.perf_counter()

        audio = Generate_Speech(
            text=sentence,
            voice="af_sky"
        )

        tts_time = (
            time.perf_counter() - tts_start
        )
        query_metrics["tts_generation_total"] += tts_time

        print(
            f"🔊 TTS generation: "
            f"{tts_time:.3f}s",
            flush=True
        )

        # -----------------------------------------
        # Measure original silence
        # -----------------------------------------

        leading, trailing = get_silence_duration(audio)

        print(
            f"🎧 Before trim: "
            f"{len(audio) / 24000:.3f}s | "
            f"leading: {leading:.3f}s | "
            f"trailing: {trailing:.3f}s",
            flush=True
        )

        # -----------------------------------------
        # Remove leading/trailing silence
        # -----------------------------------------

        audio = trim_silence(audio)

        if audio is None:
            print(
                "⚠️ Skipping silent audio",
                flush=True
            )

        else:
            print(
                f"🎧 After trim: "
                f"{len(audio) / 24000:.3f}s",
                flush=True
            )
            audio_duration = len(audio) / 24000
            query_metrics["audio_duration"] += audio_duration

            tts_audio_queue.put(audio)

        tts_queue.task_done()

def playback_worker():

    print(
        "🔊 Audio playback stream starting...",
        flush=True
    )

    with sd.OutputStream(
        samplerate=24000,
        channels=1,
        dtype="float32"
    ) as stream:

        print(
            "🟢 Audio playback stream ready",
            flush=True
        )

        while True:

            audio = tts_audio_queue.get()

            if audio is None:
                query_metrics["last_audio_finished"] = time.perf_counter()
                response_total = (query_metrics["last_audio_finished"] - query_metrics["query_start"])
                print("\n" + "=" * 50, flush=True)
                print(
                    f"📊 RESPONSE METRICS",
                    flush=True
                )

                print(
                    f"⚡ Retrieval: "
                    f"{query_metrics['retrieval_time']:.3f}s",
                    flush=True
                )

                print(
                    f"🤖 LLM: "
                    f"{query_metrics['llm_time']:.3f}s",
                    flush=True
                )

                print(
                    f"🚀 TTFA: "
                    f"{query_metrics['first_audio_started'] - query_metrics['query_start']:.3f}s",
                    flush=True
                )

                print(
                    f"🔊 TTS compute: "
                    f"{query_metrics['tts_generation_total']:.3f}s",
                    flush=True
                )

                print(
                    f"🎧 Audio duration: "
                    f"{query_metrics['audio_duration']:.3f}s",
                    flush=True
                )

                print(
                    f"⏱️ Response complete: "
                    f"{response_total:.3f}s",
                    flush=True
                )

                print("=" * 50, flush=True)
                tts_audio_queue.task_done()
                break

            print(
                "🔊 Playing audio...",
                flush=True
            )

            playback_start = time.perf_counter()

            # Make sure audio is float32
            audio = np.asarray(
                audio,
                dtype=np.float32
            )

            # Make mono audio shape:
            # (samples,) → (samples, 1)
            if audio.ndim == 1:
                audio = audio.reshape(-1, 1)

            if query_metrics["first_audio_started"] is None:

                query_metrics["first_audio_started"] = (
                    time.perf_counter()
                )

                ttfa = (
                    query_metrics["first_audio_started"]
                    - query_metrics["query_start"]
                )

                print(
                    f"🚀 TTFA: {ttfa:.3f} seconds",
                    flush=True
                )
            # Write directly into the persistent
            # audio output stream
            stream.write(audio)

            playback_time = (
                time.perf_counter()
                - playback_start
            )

            print(
                f"📡 Playback: "
                f"{playback_time:.3f}s",
                flush=True
            )

            tts_audio_queue.task_done()

    print(
        "🛑 Audio playback stream stopped",
        flush=True
    )
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

                top_chunk = Retrieve_Remote(final_query)
                print("*"*50)
                print(f"Recieved Chunk: {top_chunk}")
                print("*"*50)
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
                    

                    answer = Stream_LLM_To_TTS(
                        query=final_query,
                        context=top_chunk,
                        tts_queue=tts_queue
                    )

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
    text=" Hi, I'm Edith from VisaFlow. How may I help you?",
    voice="af_sky"
)

sd.play(greeting,samplerate=24000)
sd.wait()

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