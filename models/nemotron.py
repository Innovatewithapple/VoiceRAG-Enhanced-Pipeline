import base64
import json
import queue
import threading

import numpy as np
import websocket


# =========================================================
# CONFIG
# =========================================================

WS_URL = "ws://127.0.0.1:8080/v1/realtime"

SAMPLE_RATE = 16000
CHANNELS = 1

# We already tested this successfully.
CHUNK_MS = 160

CHUNK_SAMPLES = int(
    SAMPLE_RATE * CHUNK_MS / 1000
)


# =========================================================
# NEMOTRON STREAMER
# =========================================================

class NemotronStreamer:

    def __init__(self):

        self.audio_queue = queue.Queue()

        self.stop_event = threading.Event()

        self.ws = None

        self.receiver_thread = None
        self.worker_thread = None

        # Latest completed transcript
        self.final_transcript = ""

        # Used when VAD tells Nemotron that
        # the current utterance is finished.
        self.final_event = threading.Event()

        self.connected = threading.Event()

        self.lock = threading.Lock()

    # =====================================================
    # START
    # =====================================================

    def start(self):

        self.ws = websocket.create_connection(
            WS_URL,
            timeout=5
        )
        # 5 seconds is only for establishing the connection.
        # After connecting, keep recv() blocking indefinitely.
        self.ws.settimeout(None)
        print("🟢 Connected to NeMo realtime")

        self.ws.send(
            json.dumps({
                "type": "session.update",
                "session": {
                    "sample_rate": SAMPLE_RATE,
                    "channels": CHANNELS,
                }
            })
        )

        # -------------------------------------------------
        # Receiver
        # -------------------------------------------------

        self.receiver_thread = threading.Thread(
            target=self._receive_messages,
            daemon=True
        )

        self.receiver_thread.start()

        # -------------------------------------------------
        # Audio sender
        # -------------------------------------------------

        self.worker_thread = threading.Thread(
            target=self._audio_worker,
            daemon=True
        )

        self.worker_thread.start()

        self.connected.set()

        print(
            f"🟢 Nemotron streaming worker started "
            f"({CHUNK_MS} ms chunks)"
        )

    # =====================================================
    # RECEIVE SERVER EVENTS
    # =====================================================

    def _receive_messages(self):

        try:

            while not self.stop_event.is_set():

                message = self.ws.recv()

                if not message:
                    continue

                if isinstance(message, bytes):
                    continue

                data = json.loads(message)

                event_type = data.get(
                    "type",
                    ""
                )

                # =========================================
                # SESSION CREATED
                # =========================================

                if event_type == "session.created":

                    print(
                        "🟢 NeMo realtime session created",
                        flush=True
                    )

                # =========================================
                # SESSION UPDATED
                # =========================================

                elif event_type == "session.updated":

                    print(
                        "🟢 Session configured",
                        flush=True
                    )

                # =========================================
                # PARTIAL TRANSCRIPT
                # =========================================

                elif event_type in (
                    "transcription.delta",
                    "conversation.item.input_audio_transcription.delta",
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

                # =========================================
                # FINAL TRANSCRIPT
                # =========================================

                elif event_type in (
                    "transcription.done",
                    "conversation.item.input_audio_transcription.completed",
                ):

                    text = (
                        data.get("text")
                        or data.get("transcript")
                        or ""
                    )

                    if text:

                        with self.lock:

                            self.final_transcript = (
                                text.strip()
                            )

                        print(
                            f"\n✅ Final: "
                            f"{self.final_transcript}",
                            flush=True
                        )

                    # Tell finish_utterance()
                    # that Nemotron has finished.
                    self.final_event.set()

                # =========================================
                # SPEECH START
                # =========================================

                elif event_type== "input_audio_buffer.speech_started":

                    print("\n🎤 Nemotron speech started",flush=True)

                # =========================================
                # SPEECH END
                # =========================================

                elif (
                    event_type
                    == "input_audio_buffer.speech_stopped"
                ):

                    print(
                        "\n🔇 Nemotron speech ended",
                        flush=True
                    )

                # =========================================
                # ERROR
                # =========================================

                elif event_type == "error":

                    print(
                        "\n❌ Nemotron error:",
                        data,
                        flush=True
                    )

                    self.final_event.set()

        except Exception as e:

            if not self.stop_event.is_set():
                print("\n❌ Nemotron receiver error:",e,flush=True)
                self.final_event.set()

    # =====================================================
    # ADD MICROPHONE AUDIO
    # =====================================================

    def add_audio(self, audio):

        if self.stop_event.is_set():
            return

        self.audio_queue.put(
            audio.copy()
        )

    # =====================================================
    # AUDIO WORKER
    # =====================================================

    def _audio_worker(self):

        buffer = np.empty(
            0,
            dtype=np.int16
        )

        while not self.stop_event.is_set():

            try:

                audio = self.audio_queue.get(
                    timeout=0.1
                )

            except queue.Empty:

                continue

            # ------------------------------------------------
            # float32 → PCM16
            # ------------------------------------------------

            audio = np.clip(audio,-1.0,1.0)

            pcm16 = (audio * 32767).astype(np.int16)

            buffer = np.concatenate((buffer,pcm16))

            # ------------------------------------------------
            # Send fixed-size chunks
            # ------------------------------------------------

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

                encoded = base64.b64encode(chunk.tobytes()).decode("ascii")

                try:

                    self.ws.send(
                        json.dumps({
                            "type":
                                "input_audio_buffer.append",
                            "audio":
                                encoded,
                        })
                    )

                except Exception as e:
                    print("\n❌ Nemotron send error:",e,flush=True)
                    self.stop_event.set()
                    return

    # =====================================================
    # FINISH CURRENT UTTERANCE
    # =====================================================

    def finish_utterance(self):

        if self.stop_event.is_set():
            return ""

        # -------------------------------------------------
        # Clear old final-event state
        # -------------------------------------------------

        self.final_event.clear()

        # -------------------------------------------------
        # Tell Nemotron to finalize the current audio.
        # -------------------------------------------------

        try:

            self.ws.send(
                json.dumps({
                    "type":
                        "input_audio_buffer.commit"
                })
            )

        except Exception as e:
            print("\n❌ Failed to finalize Nemotron:",e,flush=True)
            return ""

        print("\n⏳ Waiting for Nemotron final transcript...",flush=True)

        # -------------------------------------------------
        # Wait for the actual final event.
        #
        # No arbitrary sleep().
        # -------------------------------------------------

        finished = self.final_event.wait(timeout=5.0)

        if not finished:
            print("⚠️ Nemotron final transcript timeout", flush=True)
            return ""

        # -------------------------------------------------
        # Return the transcript safely.
        # -------------------------------------------------

        with self.lock:
            return self.final_transcript.strip()

    # =====================================================
    # RESET FOR NEXT UTTERANCE
    # =====================================================

    def reset_transcript(self):
        with self.lock:
            self.final_transcript = ""
        self.final_event.clear()

    # =====================================================
    # STOP
    # =====================================================

    def stop(self):
        self.stop_event.set()

        try:
            if self.ws:
                self.ws.close()

        except Exception:
            pass