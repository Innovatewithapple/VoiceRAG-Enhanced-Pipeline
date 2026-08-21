import base64
import json
import queue
import threading
import time

import numpy as np
import sounddevice as sd
import websocket


# =========================================================
# CONFIG
# =========================================================

WS_URL = "ws://127.0.0.1:8080/v1/realtime"

SAMPLE_RATE = 16000
CHANNELS = 1

TEST_DURATION = 10

# Flexible!
# Change this later to 160, 320, 560, etc.
CHUNK_MS = 160

CHUNK_SAMPLES = int(
    SAMPLE_RATE * CHUNK_MS / 1000
)


# =========================================================
# STATE
# =========================================================

audio_queue = queue.Queue()

stop_event = threading.Event()

ws = None


# =========================================================
# MICROPHONE
# =========================================================

def audio_callback(
    indata,
    frames,
    callback_time,
    status
):

    if status:
        print(
            f"⚠️ Mic: {status}",
            flush=True
        )

    if stop_event.is_set():
        return

    # Float32 mono
    audio = indata[:, 0].copy()

    # float32 [-1, 1] → PCM16
    audio = np.clip(
        audio,
        -1.0,
        1.0
    )

    pcm16 = (
        audio * 32767
    ).astype(
        np.int16
    )

    audio_queue.put(
        pcm16
    )


# =========================================================
# WEBSOCKET RECEIVER
# =========================================================

def receive_messages():

    global ws

    try:

        while not stop_event.is_set():

            message = ws.recv()

            if not message:
                continue

            # Server messages are JSON
            if isinstance(message, bytes):
                continue

            data = json.loads(message)

            event_type = data.get(
                "type",
                ""
            )

            # -----------------------------------------
            # Connection
            # -----------------------------------------

            if event_type == "session.created":

                print(
                    "🟢 NeMo realtime session created",
                    flush=True
                )

            # -----------------------------------------
            # Session
            # -----------------------------------------

            elif event_type == "session.updated":

                print(
                    "🟢 Session configured",
                    flush=True
                )

            # -----------------------------------------
            # Partial transcript
            # -----------------------------------------

            elif (
                event_type
                in (
                    "transcription.delta",
                    "conversation.item.input_audio_transcription.delta",
                )
            ):

                text = (
                    data.get("delta")
                    or data.get("text")
                    or ""
                )

                if text:
                    print(
                        text,
                        end="",
                        flush=True
                    )

            # -----------------------------------------
            # Final transcript
            # -----------------------------------------

            elif (
                event_type
                in (
                    "transcription.done",
                    "conversation.item.input_audio_transcription.completed",
                )
            ):

                text = (
                    data.get("text")
                    or data.get("transcript")
                    or ""
                )

                if text:

                    print(
                        f"\n✅ Final: {text}",
                        flush=True
                    )

            # -----------------------------------------
            # Speech started
            # -----------------------------------------

            elif (
                event_type
                == "input_audio_buffer.speech_started"
            ):

                print(
                    "\n🎤 Speech detected: ",
                    end="",
                    flush=True
                )

            # -----------------------------------------
            # Speech stopped
            # -----------------------------------------

            elif (
                event_type
                == "input_audio_buffer.speech_stopped"
            ):

                print(
                    "\n🔇 Speech ended",
                    flush=True
                )

            # -----------------------------------------
            # Errors
            # -----------------------------------------

            elif event_type == "error":

                print(
                    "\n❌ NeMo error:",
                    data,
                    flush=True
                )

            # -----------------------------------------
            # DEBUG
            # -----------------------------------------

            else:

                # We intentionally don't print
                # every server event.
                pass

    except Exception as e:

        if not stop_event.is_set():

            print(
                "\n❌ WebSocket receiver:",
                e,
                flush=True
            )

            stop_event.set()


# =========================================================
# NEMOTRON WORKER
# =========================================================

def run_nemotron():

    global ws

    print(
        "🔌 Connecting to NeMo...",
        flush=True
    )

    try:

        ws = websocket.create_connection(
            WS_URL,
            timeout=5
        )

        print(
            "🟢 Connected to NeMo realtime",
            flush=True
        )

        # -------------------------------------------------
        # Tell server about our stream.
        # -------------------------------------------------

        ws.send(
            json.dumps({
                "type": "session.update",
                "session": {
                    "sample_rate": SAMPLE_RATE,
                    "channels": CHANNELS,
                }
            })
        )

        # -------------------------------------------------
        # Start receiver thread.
        # -------------------------------------------------

        receiver = threading.Thread(
            target=receive_messages,
            daemon=True
        )

        receiver.start()

        print(
            f"📦 Chunk size: "
            f"{CHUNK_MS} ms "
            f"({CHUNK_SAMPLES} samples)",
            flush=True
        )

        print(
            "🟢 Nemotron streaming worker started",
            flush=True
        )

        # -------------------------------------------------
        # Accumulate microphone samples
        # -------------------------------------------------

        buffer = np.empty(
            0,
            dtype=np.int16
        )

        while not stop_event.is_set():

            try:

                audio = audio_queue.get(
                    timeout=0.1
                )

            except queue.Empty:

                continue

            buffer = np.concatenate(
                (
                    buffer,
                    audio
                )
            )

            # -------------------------------------------------
            # Send fixed-size chunks
            # -------------------------------------------------

            while (
                len(buffer)
                >= CHUNK_SAMPLES
            ):

                chunk = buffer[
                    :CHUNK_SAMPLES
                ]

                buffer = buffer[
                    CHUNK_SAMPLES:
                ]

                # PCM16 → base64
                audio_b64 = base64.b64encode(
                    chunk.tobytes()
                ).decode(
                    "ascii"
                )

                ws.send(
                    json.dumps({
                        "type":
                            "input_audio_buffer.append",
                        "audio":
                            audio_b64,
                    })
                )

        # -------------------------------------------------
        # Send remaining audio
        # -------------------------------------------------

        if len(buffer) > 0:

            audio_b64 = base64.b64encode(
                buffer.tobytes()
            ).decode(
                "ascii"
            )

            ws.send(
                json.dumps({
                    "type":
                        "input_audio_buffer.append",
                    "audio":
                        audio_b64,
                })
            )

        # -------------------------------------------------
        # Finalize stream
        # -------------------------------------------------

        try:

            ws.send(
                json.dumps({
                    "type":
                        "input_audio_buffer.commit",
                    "final":
                        True,
                })
            )

        except Exception:
            pass

        time.sleep(0.5)

    except Exception as e:

        print(
            "\n❌ Nemotron worker error:",
            e,
            flush=True
        )

    finally:

        try:

            if ws:
                ws.close()

        except Exception:
            pass


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print(
        "🚀 Nemotron Live Microphone Test"
    )
    print(
        "=" * 60
    )
    print(
        f"⏱️ Duration: {TEST_DURATION} seconds"
    )
    print(
        f"🎧 Sample rate: {SAMPLE_RATE}"
    )
    print(
        f"📦 Chunk: {CHUNK_MS} ms"
    )
    print()

    # -----------------------------------------------------
    # Start Nemotron worker
    # -----------------------------------------------------

    worker = threading.Thread(
        target=run_nemotron,
        daemon=True
    )

    worker.start()

    # Give connection a moment
    time.sleep(0.5)

    print(
        "\n🎤 Starting microphone..."
    )

    print(
        "🗣️ Speak now!"
    )

    print(
        "⏳ Automatically stopping in "
        f"{TEST_DURATION} seconds..."
    )

    print(
        "-" * 60
    )

    start_time = time.monotonic()

    try:

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=512,
            callback=audio_callback,
        ):

            while (
                time.monotonic()
                - start_time
                < TEST_DURATION
            ):

                time.sleep(0.05)

    except Exception as e:

        print(
            "\n❌ Microphone error:",
            e
        )

    finally:

        stop_event.set()

        # Give worker a moment to finish
        worker.join(
            timeout=1.0
        )

        print()
        print(
            "-" * 60
        )
        print(
            "⏹️ 10-second test finished"
        )


# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":
    main()