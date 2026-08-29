import time

import numpy as np
import sounddevice as sd

from models.tts import Generate_Speech

from audio.audio_state import (
    tts_queue,
    tts_audio_queue,
    query_metrics
)

from audio.audio_utils import (
    get_silence_duration,
    trim_silence
)


def tts_worker():

    while True:

        sentence = tts_queue.get()

        if sentence is None:

            # Tell playback that this query is finished
            tts_audio_queue.put(None)

            tts_queue.task_done()

            # Worker stays alive for the next query
            continue

        print(
            f"\n🔊 Speaking: {sentence}",
            flush=True
        )

        tts_start = time.perf_counter()

        audio = Generate_Speech(
            text=sentence,
            voice="am_michael"
        )

        tts_time = (
            time.perf_counter()
            - tts_start
        )

        query_metrics[
            "tts_generation_total"
        ] += tts_time

        print(
            f"🔊 TTS generation: "
            f"{tts_time:.3f}s",
            flush=True
        )

        # -----------------------------------------
        # Measure original silence
        # -----------------------------------------

        leading, trailing = (
            get_silence_duration(audio)
        )

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

            audio_duration = (
                len(audio) / 24000
            )

            query_metrics[
                "audio_duration"
            ] += audio_duration

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

                query_metrics[
                    "last_audio_finished"
                ] = time.perf_counter()

                response_total = (
                    query_metrics[
                        "last_audio_finished"
                    ]
                    - query_metrics[
                        "query_start"
                    ]
                )

                print(
                    "\n" + "=" * 50,
                    flush=True
                )

                print(
                    "📊 RESPONSE METRICS",
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

                print(
                    "=" * 50,
                    flush=True
                )

                tts_audio_queue.task_done()

                # Keep worker alive for next query
                continue

            print(
                "🔊 Playing audio...",
                flush=True
            )

            playback_start = time.perf_counter()

            audio = np.asarray(
                audio,
                dtype=np.float32
            )

            if audio.ndim == 1:
                audio = audio.reshape(-1, 1)

            if query_metrics[
                "first_audio_started"
            ] is None:

                query_metrics[
                    "first_audio_started"
                ] = time.perf_counter()

                ttfa = (
                    query_metrics[
                        "first_audio_started"
                    ]
                    - query_metrics[
                        "query_start"
                    ]
                )

                print(
                    f"🚀 TTFA: "
                    f"{ttfa:.3f} seconds",
                    flush=True
                )

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