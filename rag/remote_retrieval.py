import asyncio
import json
import queue
import threading
import time

import websockets


# =========================================================
# COLAB RETRIEVAL + RERANKING WEBSOCKET
# =========================================================

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

        # Only one RAG request at a time.
        # This prevents responses from different queries
        # from getting mixed together.
        self.request_lock = threading.Lock()

        self.start()

    # =====================================================
    # START PERSISTENT EVENT LOOP
    # =====================================================

    def start(self):

        self.thread = threading.Thread(
            target=self._run_loop,
            daemon=True
        )

        self.thread.start()

        if not self.connected.wait(timeout=30):

            raise RuntimeError(
                "❌ Remote Retrieval WebSocket connection timeout."
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
            "🔌 Connecting to Remote Retrieval WebSocket...",
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
            "🟢 Remote Retrieval WebSocket connected.",
            flush=True
        )

    # =====================================================
    # ONE REQUEST
    # =====================================================

    async def _request_async(
        self,
        query,
        output_queue
    ):

        try:

            await self.websocket.send(
                json.dumps({
                    "query": query
                })
            )

            # ---------------------------------------------
            # Wait for Colab response
            # ---------------------------------------------

            while True:

                message = await self.websocket.recv()

                data = json.loads(message)

                output_queue.put(data)

                # RAG server sends exactly one response
                # for each query, so we can stop here.
                break

        except Exception as e:

            output_queue.put({
                "error": str(e)
            })

    # =====================================================
    # REQUEST
    # =====================================================

    def request(self, query):

        with self.request_lock:

            if not self.connected.is_set():

                print(
                    "⚠️ Remote Retrieval WebSocket "
                    "is not connected.",
                    flush=True
                )

                return None

            output_queue = queue.Queue()

            asyncio.run_coroutine_threadsafe(

                self._request_async(
                    query,
                    output_queue
                ),

                self.loop
            )

            data = output_queue.get()

            if "error" in data:

                print(
                    f"⚠️ Remote retrieval WebSocket "
                    f"error: {data['error']}",
                    flush=True
                )

                return None

            return data

    # =====================================================
    # CLOSE
    # =====================================================

    def close(self):

        if self.stopped.is_set():

            return

        self.stopped.set()

        print(
            "🔴 Closing Remote Retrieval WebSocket...",
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
            "🔴 Remote Retrieval WebSocket closed.",
            flush=True
        )


# =========================================================
# GLOBAL CLIENT
# =========================================================

retrieval_client = None


# =========================================================
# PUBLIC FUNCTION
# =========================================================

def Retrieve_Remote(
    query,
    timeout=30
):

    if retrieval_client is None:

        print(
            "❌ Remote Retrieval WebSocket client "
            "has not been initialized.",
            flush=True
        )

        return None

    request_start = time.perf_counter()

    try:

        response = retrieval_client.request(
            query
        )

        total_time = (
            time.perf_counter()
            - request_start
        )

        print(
            f"🌐 Remote Retrieval WebSocket: "
            f"{total_time:.3f}s",
            flush=True
        )

        if response is None:

            return None

        # -------------------------------------------------
        # Print timing returned by Colab
        # -------------------------------------------------

        if "retrieval_time" in response:

            print(
                f"📚 Retrieval: "
                f"{response['retrieval_time']:.3f}s",
                flush=True
            )

        if "reranking_time" in response:

            print(
                f"🔄 Reranking: "
                f"{response['reranking_time']:.3f}s",
                flush=True
            )

        if "total_time" in response:

            print(
                f"⚡ Colab RAG total: "
                f"{response['total_time']:.3f}s",
                flush=True
            )

        return response

    except Exception as e:

        print(
            f"⚠️ Remote retrieval failed: {e}",
            flush=True
        )

        return None