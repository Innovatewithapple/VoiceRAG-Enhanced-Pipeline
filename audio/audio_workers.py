import time
import numpy as np
import sounddevice as sd
from models.tts import Generate_Speech
import audio.audio_state as audio_state
import wave
import os

RECORD_DEMO_AUDIO = True

DEMO_AUDIO_PATH = "demo_ai.wav"

from audio.audio_state import (
    tts_queue,
    tts_audio_queue,
    query_metrics,
    tts_interrupt_event,
    tts_speaking_event
)

from audio.audio_utils import (
    get_silence_duration,
    trim_silence
)


# =========================================================
# TTS WORKER
# =========================================================

def tts_worker():
    demo_wav = None

    if RECORD_DEMO_AUDIO:

        demo_wav = wave.open(
            DEMO_AUDIO_PATH,
            "wb"
        )

        demo_wav.setnchannels(1)
        demo_wav.setsampwidth(2)
        demo_wav.setframerate(24000)

        print(
            f"🎙️ Demo AI recording → {DEMO_AUDIO_PATH}",
            flush=True
        )

    while True:

        sentence = tts_queue.get()

        if sentence is None:

            # Tell playback that this query is finished
            tts_audio_queue.put(None)

            tts_queue.task_done()

            # Worker stays alive
            continue

        # =========================================================
        # CAPTURE CURRENT TTS GENERATION
        # =========================================================

        generation_id = audio_state.tts_generation_id

        print(
            f"🧾 TTS generation: {generation_id}",
            flush=True
        )

        print(
            f"\n🔊 Speaking: {sentence}",
            flush=True
        )

        tts_start = time.perf_counter()

        audio = Generate_Speech(
            text=sentence,
            voice="af_sarah"
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
        # Remove silence
        # -----------------------------------------

        audio = trim_silence(audio)
        if (
            RECORD_DEMO_AUDIO
            and demo_wav is not None
            and audio is not None
        ):

            audio_int16 = (
                np.clip(audio, -1.0, 1.0)
                * 32767
            ).astype(np.int16)

            demo_wav.writeframes(
                audio_int16.tobytes()
            )

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

            tts_audio_queue.put(
                (
                    generation_id,
                    audio
                )
            )

        tts_queue.task_done()

        


# =========================================================
# PLAYBACK WORKER
# =========================================================

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

            # =================================================
            # GET NEXT AUDIO ITEM
            # =================================================

            item = tts_audio_queue.get()

            # =================================================
            # QUERY FINISHED
            # =================================================

            if item is None:

                tts_speaking_event.clear()

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

                # if (
                #     query_metrics[
                #         "first_audio_started"
                #     ] is not None
                # ):

                #     print(
                #         f"🚀 TTFA: "
                #         f"{query_metrics['first_audio_started'] - query_metrics['query_start']:.3f}s",
                #         flush=True
                #     )

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

                continue

            # =================================================
            # GET GENERATION ID + AUDIO
            # =================================================

            generation_id, audio = item

            # =================================================
            # CHECK WHETHER AUDIO IS STALE
            # =================================================

            if (
                generation_id
                != audio_state.tts_generation_id
            ):

                print(
                    f"🗑️ Discarding stale TTS audio "
                    f"(generation {generation_id} != "
                    f"current {audio_state.tts_generation_id})",
                    flush=True
                )

                tts_audio_queue.task_done()

                continue

            # =================================================
            # PREPARE AUDIO
            # =================================================

            print(
                "🔊 Playing audio...",
                flush=True
            )

            audio = np.asarray(
                audio,
                dtype=np.float32
            )

            if audio.ndim == 1:

                audio = audio.reshape(-1, 1)

            # =================================================
            # TTS IS NOW ACTUALLY SPEAKING
            # =================================================

            tts_speaking_event.set()

            print(
                "🟢 TTS SPEAKING EVENT = SET",
                flush=True
            )

            # =================================================
            # TTFA
            # =================================================

            if (
                query_metrics[
                    "first_audio_started"
                ] is None
            ):

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

                # print(
                #     f"🚀 TTFA: "
                #     f"{ttfa:.3f} seconds",
                #     flush=True
                # )

            # =================================================
            # PLAY AUDIO IN SMALL BLOCKS
            # =================================================

            playback_start = time.perf_counter()

            block_size = 1024

            interrupted = False

            for start_index in range(
                0,
                len(audio),
                block_size
            ):

                # =================================================
                # CHECK INTERRUPTION
                # =================================================

                if tts_interrupt_event.is_set():

                    print(
                        "🛑 INTERRUPTION EVENT DETECTED BY PLAYBACK",
                        flush=True
                    )

                    print(
                        "🛑 STOPPING CURRENT TTS AUDIO",
                        flush=True
                    )

                    interrupted = True

                    break

                # =================================================
                # GET NEXT AUDIO BLOCK
                # =================================================

                end_index = min(
                    start_index + block_size,
                    len(audio)
                )

                block = audio[
                    start_index:end_index
                ]


                stream.write(block)

            # =================================================
            # INTERRUPTION CLEANUP
            # =================================================

            if interrupted:

                print(
                    "🔴 TTS PLAYBACK STOPPED",
                    flush=True
                )

                # -------------------------------------------------
                # Stop current audio output
                # -------------------------------------------------

                try:

                    stream.stop()
                    stream.start()

                except Exception as e:

                    print(
                        f"⚠️ Could not restart audio stream: "
                        f"{e}",
                        flush=True
                    )

                # -------------------------------------------------
                # Clear speaking state
                # -------------------------------------------------

                tts_speaking_event.clear()

                # -------------------------------------------------
                # Clear interruption event
                # -------------------------------------------------

                tts_interrupt_event.clear()

                # -------------------------------------------------
                # Current audio item is finished/aborted
                # -------------------------------------------------

                tts_audio_queue.task_done()

                # -------------------------------------------------
                # DO NOT PLAY THE REST
                # -------------------------------------------------

                continue

            # =================================================
            # NORMAL PLAYBACK FINISHED
            # =================================================

            playback_time = (
                time.perf_counter()
                - playback_start
            )

            print(
                f"📡 Playback: "
                f"{playback_time:.3f}s",
                flush=True
            )

            tts_speaking_event.clear()

            tts_audio_queue.task_done()

    print(
        "🛑 Audio playback stream stopped",
        flush=True
    )