import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

import asyncio
import json
import time

import numpy as np
import sounddevice as sd
import websockets

from models.tts import Generate_Speech


# =========================================================
# CONFIG
# =========================================================

QWEN_URL = "wss://absentee-mulled-stadium.ngrok-free.dev/ws"

MODEL = "Qwen3-30B-A3B-Q4_K_M.gguf"

VOICE = "af_sarah"

SAMPLE_RATE = 24000

# Very short acknowledgement
MAX_TOKENS = 40


# =========================================================
# QUICK ACKNOWLEDGEMENT PROMPT
# =========================================================

SYSTEM_PROMPT = """
You are the conversational bridge for a real-time telephone voice assistant.

Your only purpose is to keep the conversation natural while the main
assistant processes the user's request in the background.

Do NOT answer the user's question.
Do NOT provide factual information.
Do NOT solve or explain the request.
Do NOT claim that you have checked, verified, retrieved, or completed anything.

Based only on the user's latest utterance, produce a very brief,
natural acknowledgement suitable for a telephone conversation.

The response must:

- be very short
- normally be one brief spoken line
- acknowledge the user's intent naturally
- never answer the actual question
- never ask the user a question
- never request additional information
- never provide factual information
- never explain anything
- never contain multiple conversational turns
- sound natural when spoken aloud

Generate ONLY the acknowledgement.
"""


# =========================================================
# TEST QUERIES
# =========================================================

QUERIES = [

    "Hi, is this VisaFlow? Am I talking to the right person?",

    "Hi, my name is Mihir, and I want to know about the cancellation policy.",

    "I've been waiting for my application for two weeks.",

    "I just wanted to know if you help with visa applications.",

    "Can you tell me what documents I need for a student visa?"

]


# =========================================================
# TIMESTAMP HELPER
# =========================================================

def timestamp(start):

    return time.perf_counter() - start


# =========================================================
# GENERATE ONE AUDIO
# =========================================================
#
# IMPORTANT:
#
# This function ONLY GENERATES audio.
#
# It does NOT play anything.
#
# Therefore multiple instances of this function can be
# running while Qwen continues streaming.
#
# =========================================================

async def generate_audio(
    sentence,
    query_start,
    split_time,
    chunk_number
):

    # -----------------------------------------------------
    # TTS GENERATION START
    # -----------------------------------------------------

    generation_start = time.perf_counter()

    print(
        "\n"
        + "-" * 60,
        flush=True
    )

    print(
        f"🎙️ TTS GENERATION #{chunk_number} START",
        flush=True
    )

    print(
        f"🗣️ Text: {sentence}",
        flush=True
    )

    print(
        f"⏱️ Split → generation: "
        f"{generation_start - split_time:.3f}s",
        flush=True
    )

    print(
        f"⏱️ From query: "
        f"{generation_start - query_start:.3f}s",
        flush=True
    )

    # -----------------------------------------------------
    # KOKORO
    # -----------------------------------------------------

    audio = await asyncio.to_thread(

        Generate_Speech,

        text=sentence,

        voice=VOICE

    )

    # -----------------------------------------------------
    # GENERATION COMPLETE
    # -----------------------------------------------------

    generation_end = time.perf_counter()

    generation_time = (
        generation_end
        - generation_start
    )

    print(
        f"🎵 TTS GENERATION #{chunk_number} READY",
        flush=True
    )

    print(
        f"🔊 Kokoro generation: "
        f"{generation_time:.3f}s",
        flush=True
    )

    print(
        f"⏱️ Audio ready from query: "
        f"{generation_end - query_start:.3f}s",
        flush=True
    )

    # -----------------------------------------------------
    # PREPARE AUDIO
    # -----------------------------------------------------

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    if audio.ndim == 1:

        audio = audio.reshape(
            -1,
            1
        )

    audio_duration = (
        len(audio)
        / SAMPLE_RATE
    )

    print(
        f"🎧 Audio duration: "
        f"{audio_duration:.3f}s",
        flush=True
    )

    # -----------------------------------------------------
    # RETURN EVERYTHING NEEDED BY PLAYBACK
    # -----------------------------------------------------

    return {

        "chunk_number":
            chunk_number,

        "sentence":
            sentence,

        "audio":
            audio,

        "generation_start":
            generation_start,

        "generation_end":
            generation_end,

        "generation_time":
            generation_time,

        "audio_duration":
            audio_duration,

        "query_start":
            query_start

    }


# =========================================================
# PLAYBACK ONE AUDIO
# =========================================================
#
# This function ONLY plays audio.
#
# Playback is called sequentially by the playback worker.
#
# =========================================================

def play_audio(
    result,
    query_start,
    first_audio_state
):

    chunk_number = result[
        "chunk_number"
    ]

    sentence = result[
        "sentence"
    ]

    audio = result[
        "audio"
    ]

    # -----------------------------------------------------
    # PLAYBACK START
    # -----------------------------------------------------

    playback_start = time.perf_counter()

    print(
        "\n"
        + "🔊" * 10,
        flush=True
    )

    print(
        f"🔊 PLAYBACK #{chunk_number} START",
        flush=True
    )

    print(
        f"🗣️ {sentence}",
        flush=True
    )

    print(
        f"⏱️ From query: "
        f"{playback_start - query_start:.3f}s",
        flush=True
    )

    # -----------------------------------------------------
    # FIRST AUDIO
    # -----------------------------------------------------

    if first_audio_state["value"]:

        first_audio_state["value"] = False

        ttfa = (
            playback_start
            - query_start
        )

        first_audio_state[
            "ttfa"
        ] = ttfa

        print(
            "\n"
            + "🔥" * 20,
            flush=True
        )

        print(
            f"🔥 QUICK RESPONSE TTFA: "
            f"{ttfa:.3f}s",
            flush=True
        )

        print(
            "🔥" * 20,
            flush=True
        )

    # -----------------------------------------------------
    # PLAY
    # -----------------------------------------------------

    sd.play(
        audio,
        samplerate=SAMPLE_RATE
    )

    # -----------------------------------------------------
    # WAIT UNTIL THIS AUDIO FINISHES
    # -----------------------------------------------------

    sd.wait()

    # -----------------------------------------------------
    # PLAYBACK COMPLETE
    # -----------------------------------------------------

    playback_end = time.perf_counter()

    print(
        f"🎧 PLAYBACK #{chunk_number} FINISHED",
        flush=True
    )

    print(
        f"⏱️ Playback duration: "
        f"{playback_end - playback_start:.3f}s",
        flush=True
    )

    print(
        f"⏱️ Finished from query: "
        f"{playback_end - query_start:.3f}s",
        flush=True
    )

    return {

        "chunk_number":
            chunk_number,

        "playback_start":
            playback_start,

        "playback_end":
            playback_end,

        "playback_duration":
            playback_end
            - playback_start

    }


# =========================================================
# SERIAL PLAYBACK WORKER
# =========================================================
#
# THIS IS THE IMPORTANT PART.
#
# Generation can happen concurrently.
#
# Playback ALWAYS happens:
#
# chunk 1
# chunk 2
# chunk 3
# ...
#
# Never simultaneously.
#
# =========================================================

async def playback_worker(
    playback_queue,
    query_start,
    first_audio_state
):

    while True:

        item = await playback_queue.get()

        # =================================================
        # END OF QUERY
        # =================================================

        if item is None:

            playback_queue.task_done()

            print(
                "\n🟢 Playback queue completely finished.",
                flush=True
            )

            return

        # =================================================
        # PLAY THIS AUDIO
        # =================================================

        result = await asyncio.to_thread(

            play_audio,

            item,

            query_start,

            first_audio_state

        )

        # Save playback metrics
        item["playback_start"] = result[
            "playback_start"
        ]

        item["playback_end"] = result[
            "playback_end"
        ]

        item["playback_duration"] = result[
            "playback_duration"
        ]

        playback_queue.task_done()


# =========================================================
# STREAM SPLITTER
# =========================================================

def extract_splits(buffer):

    splits = []

    while True:

        split_position = None

        # -------------------------------------------------
        # SAME SPLITTER IDEA AS REAL PIPELINE
        # -------------------------------------------------

        for punctuation in [
            ",",
            ".",
            "?",
            "!"
        ]:

            position = buffer.find(
                punctuation
            )

            if position == -1:

                continue

            if (
                split_position is None
                or position < split_position
            ):

                split_position = position

        # -------------------------------------------------
        # NOTHING READY
        # -------------------------------------------------

        if split_position is None:

            break

        # -------------------------------------------------
        # EXTRACT TEXT
        # -------------------------------------------------

        sentence = (
            buffer[
                :split_position + 1
            ]
            .strip()
        )

        # -------------------------------------------------
        # REMOVE FROM BUFFER
        # -------------------------------------------------

        buffer = (
            buffer[
                split_position + 1:
            ]
            .lstrip()
        )

        if sentence:

            splits.append(
                sentence
            )

    return buffer, splits


# =========================================================
# RUN ONE QUERY
# =========================================================

async def run_query(
    ws,
    query,
    query_number
):

    print("\n")
    print("=" * 70)
    print(
        f"🧪 QUERY {query_number}"
    )
    print("=" * 70)

    print(
        f"👤 User: {query}",
        flush=True
    )

    # =====================================================
    # PROMPT
    # =====================================================

    messages = [

        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },

        {
            "role": "user",
            "content": query
        }

    ]

    # =====================================================
    # QWEN PAYLOAD
    # =====================================================

    payload = {

        "model": MODEL,

        "messages": messages,

        "temperature": 0.7,

        "max_tokens": MAX_TOKENS,

        "top_p": 0.8,

        "stream": True,

        "top_k": 20,

        "min_p": 0,

        "chat_template_kwargs": {
            "enable_thinking": False
        }

    }

    # =====================================================
    # T0 — QUERY START
    # =====================================================

    query_start = time.perf_counter()

    await ws.send(
        json.dumps(payload)
    )

    print(
        "\n📤 T0 → QUERY SENT",
        flush=True
    )

    print(
        f"⏱️ T0 timestamp: "
        f"{timestamp(query_start):.3f}s",
        flush=True
    )

    # =====================================================
    # TTS GENERATION TASKS
    # =====================================================
    #
    # These can run concurrently.
    #
    # =====================================================

    generation_tasks = []

    # =====================================================
    # PLAYBACK QUEUE
    # =====================================================
    #
    # This queue receives COMPLETED audio.
    #
    # Playback worker consumes them sequentially.
    #
    # =====================================================

    playback_queue = asyncio.Queue()

    # =====================================================
    # FIRST AUDIO STATE
    # =====================================================

    first_audio_state = {

        "value": True,

        "ttfa": None

    }

    # =====================================================
    # START ONE PLAYBACK WORKER
    # =====================================================

    playback_task = asyncio.create_task(

        playback_worker(

            playback_queue,

            query_start,

            first_audio_state

        )

    )

    # =====================================================
    # STREAM STATE
    # =====================================================

    buffer = ""

    full_response = ""

    first_token_time = None

    first_split_time = None

    qwen_done_time = None

    chunk_number = 0

    # =====================================================
    # QWEN STREAM
    # =====================================================

    while True:

        message = await ws.recv()

        data = json.loads(
            message
        )

        # =================================================
        # ERROR
        # =================================================

        if "error" in data:

            print(
                f"\n❌ Qwen error: "
                f"{data['error']}",
                flush=True
            )

            break

        # =================================================
        # CONTENT
        # =================================================

        if "content" in data:

            text = data["content"]

            if not text:

                continue

            # ---------------------------------------------
            # T1 — FIRST TOKEN
            # ---------------------------------------------

            if first_token_time is None:

                first_token_time = (
                    time.perf_counter()
                )

                print(
                    f"\n🚀 T1 → FIRST QWEN TOKEN: "
                    f"{first_token_time - query_start:.3f}s",
                    flush=True
                )

                print(
                    "🤖 Qwen stream: ",
                    end="",
                    flush=True
                )

            # ---------------------------------------------
            # SHOW STREAM
            # ---------------------------------------------

            print(
                text,
                end="",
                flush=True
            )

            # ---------------------------------------------
            # SAVE RESPONSE
            # ---------------------------------------------

            full_response += text

            # ---------------------------------------------
            # BUFFER
            # ---------------------------------------------

            buffer += text

            # ---------------------------------------------
            # EXTRACT SPLITS
            # ---------------------------------------------

            buffer, splits = (
                extract_splits(
                    buffer
                )
            )

            # ---------------------------------------------
            # EACH SPLIT
            # ---------------------------------------------

            for sentence in splits:

                chunk_number += 1

                split_time = (
                    time.perf_counter()
                )

                # -----------------------------------------
                # FIRST SPLIT
                # -----------------------------------------

                if first_split_time is None:

                    first_split_time = (
                        split_time
                    )

                    print(
                        "\n\n"
                        f"🧾 T2 → FIRST SPLIT: "
                        f"{split_time - query_start:.3f}s",
                        flush=True
                    )

                else:

                    print(
                        "\n"
                        f"🧾 TTS SPLIT #{chunk_number}: "
                        f"{split_time - query_start:.3f}s",
                        flush=True
                    )

                print(
                    f"🗣️ {sentence}",
                    flush=True
                )

                # -----------------------------------------
                # START KOKORO GENERATION
                #
                # IMPORTANT:
                #
                # We DO NOT await it here.
                #
                # Therefore Qwen can continue streaming
                # while Kokoro generates this audio.
                # -----------------------------------------

                task = asyncio.create_task(

                    generate_audio(

                        sentence,

                        query_start,

                        split_time,

                        chunk_number

                    )

                )

                generation_tasks.append(
                    task
                )

        # =================================================
        # QWEN DONE
        # =================================================

        if data.get("done"):

            qwen_done_time = (
                time.perf_counter()
            )

            print(
                "\n\n"
                f"🏁 T7 → QWEN COMPLETE: "
                f"{qwen_done_time - query_start:.3f}s",
                flush=True
            )

            break

    # =====================================================
    # FLUSH REMAINING BUFFER
    # =====================================================

    if buffer.strip():

        chunk_number += 1

        final_text = (
            buffer.strip()
        )

        final_split_time = (
            time.perf_counter()
        )

        print(
            "\n"
            f"🧾 FINAL SPLIT #{chunk_number}: "
            f"{final_text}",
            flush=True
        )

        task = asyncio.create_task(

            generate_audio(

                final_text,

                query_start,

                final_split_time,

                chunk_number

            )

        )

        generation_tasks.append(
            task
        )

    # =====================================================
    # WAIT FOR ALL KOKORO GENERATION
    # =====================================================
    #
    # IMPORTANT:
    #
    # This waits for generation ONLY.
    #
    # Playback still has to happen.
    #
    # =====================================================

    print(
        "\n⏳ Waiting for all TTS generations...",
        flush=True
    )

    if generation_tasks:

        generated_audio = await asyncio.gather(

            *generation_tasks,

            return_exceptions=True

        )

    else:

        generated_audio = []

    # =====================================================
    # INSERT GENERATED AUDIO INTO PLAYBACK QUEUE
    # =====================================================
    #
    # IMPORTANT:
    #
    # We insert them in CHUNK ORDER.
    #
    # So even if:
    #
    # Chunk 3 generated first
    #
    # it will NOT play before Chunk 1.
    #
    # =====================================================

    print(
        "\n📥 Sending generated audio to playback queue...",
        flush=True
    )

    for result in generated_audio:

        if isinstance(
            result,
            Exception
        ):

            print(
                f"❌ TTS generation failed: "
                f"{result}",
                flush=True
            )

            continue

        await playback_queue.put(
            result
        )

    # =====================================================
    # TELL PLAYBACK WORKER:
    #
    # NO MORE AUDIO FOR THIS QUERY
    # =====================================================

    await playback_queue.put(
        None
    )

    # =====================================================
    # WAIT FOR ALL PLAYBACK
    # =====================================================
    #
    # THIS IS THE CRITICAL QUERY BOUNDARY.
    #
    # Query N+1 does not start until this completes.
    #
    # =====================================================

    print(
        "\n⏳ Waiting for ALL Query audio playback...",
        flush=True
    )

    await playback_task

    # =====================================================
    # QUERY FINISHED
    # =====================================================

    query_end = time.perf_counter()

    print("\n")
    print("=" * 70)
    print(
        f"📊 QUERY {query_number} COMPLETE"
    )
    print("=" * 70)

    # -----------------------------------------------------
    # TIMESTAMPS
    # -----------------------------------------------------

    print(
        f"📤 T0 Query sent:             "
        f"0.000s"
    )

    if first_token_time:

        print(
            f"🚀 T1 First Qwen token:       "
            f"{first_token_time - query_start:.3f}s"
        )

    if first_split_time:

        print(
            f"🧾 T2 First TTS split:        "
            f"{first_split_time - query_start:.3f}s"
        )

    if first_audio_state["ttfa"] is not None:

        print(
            f"🔊 T5 First playback / TTFA:  "
            f"{first_audio_state['ttfa']:.3f}s"
        )

    if qwen_done_time:

        print(
            f"🏁 T7 Qwen complete:          "
            f"{qwen_done_time - query_start:.3f}s"
        )

    print(
        f"🎧 ALL AUDIO FINISHED:        "
        f"{query_end - query_start:.3f}s"
    )

    print(
        f"📝 Final response:            "
        f"{full_response.strip()}"
    )

    print("=" * 70)

    # =====================================================
    # VERY IMPORTANT
    # =====================================================
    #
    # At this point:
    #
    # Qwen finished
    # +
    # all Kokoro generations finished
    # +
    # every generated audio chunk played
    #
    # ONLY NOW can the next query begin.
    #
    # =====================================================


# =========================================================
# MAIN
# =========================================================

async def main():

    print("=" * 70)

    print(
        "🚀 QUICK RESPONSE"
    )

    print(
        "QWEN STREAM → PARALLEL TTS GENERATION → SERIAL PLAYBACK"
    )

    print("=" * 70)

    # =====================================================
    # CONNECT ONCE
    # =====================================================

    print(
        "\n🔌 Establishing Qwen WebSocket...",
        flush=True
    )

    connection_start = time.perf_counter()

    async with websockets.connect(
        QWEN_URL
    ) as ws:

        connection_time = (
            time.perf_counter()
            - connection_start
        )

        print(
            f"🟢 WebSocket connected: "
            f"{connection_time:.3f}s",
            flush=True
        )

        print(
            "🟢 Connection remains open for all queries.",
            flush=True
        )

        # =================================================
        # QUERIES ARE STRICTLY SEQUENTIAL
        # =================================================

        for index, query in enumerate(
            QUERIES,
            start=1
        ):

            await run_query(

                ws,

                query,

                index

            )

            # ---------------------------------------------
            # THIS MEANS THE ENTIRE QUERY IS DONE.
            #
            # ALL AUDIO HAS FINISHED PLAYING.
            # ---------------------------------------------

            print(
                f"\n🟢 Query {index} fully finished.",
                flush=True
            )

            if index < len(QUERIES):

                print(
                    "🟢 Starting next query...",
                    flush=True
                )

                await asyncio.sleep(
                    0.5
                )

        # =================================================
        # ALL QUERIES COMPLETE
        # =================================================

        print("\n")

        print("=" * 70)

        print(
            "🔴 ALL QUERIES COMPLETED"
        )

        print(
            "🔴 Closing Qwen WebSocket..."
        )

        print("=" * 70)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    asyncio.run(main())