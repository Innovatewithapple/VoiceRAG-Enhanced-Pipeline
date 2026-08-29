import asyncio
import json
import queue
import threading
import time

import websockets

from audio.audio_state import tts_queue


COLAB_WS_URL = (
    "wss://plethora-registry-shrine.ngrok-free.dev/ws"
)


class RemoteRetrievalWebSocketClient:

    def __init__(self, url=COLAB_WS_URL):

        self.url = url

        self.loop = None
        self.thread = None

        self.websocket = None

        self.connected = threading.Event()
        self.stopped = threading.Event()

        self.request_lock = threading.Lock()

        self.start()

    # =====================================================
    # START BACKGROUND ASYNCIO LOOP
    # =====================================================

    def start(self):

        self.thread = threading.Thread(
            target=self._run_loop,
            daemon=True
        )

        self.thread.start()

        if not self.connected.wait(timeout=30):

            raise RuntimeError(
                "❌ Retrieval WebSocket connection timeout."
            )

    # =====================================================
    # ASYNCIO THREAD
    # =====================================================

    def _run_loop(self):

        self.loop = asyncio.new_event_loop()

        asyncio.set_event_loop(
            self.loop
        )

        self.loop.run_until_complete(
            self._connect()
        )

        self.loop.run_forever()

    # =====================================================
    # CONNECT
    # =====================================================

    async def _connect(self):

        print(
            "🔌 Connecting to Colab Retrieval WebSocket...",
            flush=True
        )

        self.websocket = await websockets.connect(

            self.url,

            ping_interval=20,

            ping_timeout=60,

            max_size=None
        )

        self.connected.set()

        print(
            "🟢 Colab Retrieval WebSocket connected.",
            flush=True
        )

    # =====================================================
    # QUERY
    # =====================================================

    async def _query_async(
        self,
        query,
        output_queue
    ):

        try:

            # ---------------------------------------------
            # SEND QUERY
            # ---------------------------------------------

            await self.websocket.send(
                json.dumps({
                    "query": query
                })
            )

            # ---------------------------------------------
            # RECEIVE STREAM
            # ---------------------------------------------

            while True:

                message = await self.websocket.recv()

                data = json.loads(message)

                # =========================================
                # DEBUG: WHAT EXACTLY CAME FROM COLAB?
                # =========================================

                print(
                    f"📥 FROM COLAB: "
                    f"type={data.get('type')} "
                    f"chars={len(data.get('content', ''))}",
                    flush=True
                )

                if data.get("type") == "sentence":

                    print(
                        f"📥 COLAB SENTENCE: "
                        f"{data.get('content', '')}",
                        flush=True
                    )

                # =========================================
                # ERROR
                # =========================================

                if data.get("type") == "error":

                    output_queue.put({
                        "type": "error",
                        "error": data.get("error")
                    })

                    return

                # =========================================
                # METADATA
                # =========================================

                if data.get("type") == "metadata":

                    output_queue.put({
                        "type": "metadata",
                        "data": data
                    })

                # =========================================
                # SENTENCE
                # =========================================

                elif data.get("type") == "sentence":

                    output_queue.put({
                        "type": "sentence",
                        "content": data.get(
                            "content",
                            ""
                        )
                    })

                # =========================================
                # DONE
                # =========================================

                elif data.get("type") == "done":

                    output_queue.put({
                        "type": "done"
                    })

                    return

        except Exception as e:

            output_queue.put({
                "type": "error",
                "error": str(e)
            })

    # =====================================================
    # PUBLIC QUERY
    # =====================================================

    def query(self, query):

        with self.request_lock:

            if not self.connected.is_set():

                raise RuntimeError(
                    "❌ Retrieval WebSocket is not connected."
                )

            output_queue = queue.Queue()

            future = asyncio.run_coroutine_threadsafe(

                self._query_async(
                    query,
                    output_queue
                ),

                self.loop
            )

            retrieval_time = None
            reranking_time = None
            rag_total = None

            full_response = ""

            request_start = (
                time.perf_counter()
            )
            first_stream_time = None

            while True:

                item = output_queue.get()

                item_type = item.get(
                    "type"
                )

                # =========================================
                # ERROR
                # =========================================

                if item_type == "error":

                    # Make sure the coroutine has completed
                    future.result()

                    raise RuntimeError(
                        item.get(
                            "error",
                            "Unknown WebSocket error"
                        )
                    )

                # =========================================
                # METADATA
                # =========================================

                if item_type == "metadata":

                    metadata = item["data"]

                    retrieval_time = metadata.get(
                        "retrieval_time"
                    )

                    reranking_time = metadata.get(
                        "reranking_time"
                    )

                    rag_total = metadata.get(
                        "total_time"
                    )

                    print(
                        f"📚 Retrieval: "
                        f"{retrieval_time:.3f}s",
                        flush=True
                    )

                    print(
                        f"🔄 Reranking: "
                        f"{reranking_time:.3f}s",
                        flush=True
                    )

                    print(
                        f"⚡ Colab RAG total: "
                        f"{rag_total:.3f}s",
                        flush=True
                    )

                # =========================================
                # SENTENCE
                # =========================================

                elif item_type == "sentence":

                    sentence = item.get(
                        "content",
                        ""
                    )

                    if sentence:

                        full_response += (
                            sentence + " "
                        )

                        # =========================================
                        # FIRST STREAM → TTS
                        # =========================================

                        if first_stream_time is None:

                            first_stream_time = (
                                time.perf_counter()
                                - request_start
                            )

                            print(
                                f"🚀 First stream → TTS: "
                                f"{first_stream_time:.3f}s",
                                flush=True
                            )

                        # =================================
                        # IMMEDIATELY SEND TO TTS
                        # =================================

                        print(
                            f"🔊 Sentence → TTS: "
                            f"{sentence}",
                            flush=True
                        )

                        tts_queue.put(
                            sentence
                        )

                # =========================================
                # DONE
                # =========================================

                elif item_type == "done":

                    break

            # =============================================
            # WAIT FOR COROUTINE TO FINISH
            # =============================================

            future.result()

            # =============================================
            # TELL TTS WORKER THIS QUERY IS COMPLETE
            # =============================================

            tts_queue.put(None)

            total_time = (
                time.perf_counter()
                - request_start
            )

            print(
                f"🌐 Remote Retrieval + Qwen: "
                f"{total_time:.3f}s",
                flush=True
            )

            return {

                "answer":
                    full_response.strip(),

                "retrieval_time":
                    retrieval_time,

                "reranking_time":
                    reranking_time,

                "rag_total":
                    rag_total,

                "qwen_total":
                    total_time

            }

    # =====================================================
    # CLOSE
    # =====================================================

    def close(self):

        if self.stopped.is_set():

            return

        self.stopped.set()

        print(
            "🔴 Closing Retrieval WebSocket...",
            flush=True
        )

        if self.loop:

            async def shutdown():

                try:

                    if self.websocket:

                        await self.websocket.close()

                except Exception:

                    pass

                self.loop.stop()

            asyncio.run_coroutine_threadsafe(
                shutdown(),
                self.loop
            )

        self.connected.clear()

        print(
            "🔴 Retrieval WebSocket closed.",
            flush=True
        )


# =========================================================
# GLOBAL PERSISTENT CLIENT
# =========================================================

retrieval_client = None


# =========================================================
# PUBLIC FUNCTION
# =========================================================

def Retrieve_Remote(query):

    global retrieval_client

    if retrieval_client is None:

        retrieval_client = (
            RemoteRetrievalWebSocketClient()
        )

    return retrieval_client.query(
        query
    )