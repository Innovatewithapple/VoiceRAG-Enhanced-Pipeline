import asyncio
import json
import threading
import time

import websockets


QWEN_WS_URL = (
    "wss://absentee-mulled-stadium.ngrok-free.dev/ws"
)


class QwenWebSocketClient:

    def __init__(self, url=QWEN_WS_URL):

        self.url = url

        self.loop = None
        self.thread = None

        self.websocket = None

        self.connected = threading.Event()
        self.stopped = threading.Event()

        self.request_lock = threading.Lock()

        self.start()

    # =====================================================
    # START PERSISTENT ASYNCIO THREAD
    # =====================================================

    def start(self):

        self.thread = threading.Thread(
            target=self._run_loop,
            daemon=True
        )

        self.thread.start()

        # Wait until connection is ready

        if not self.connected.wait(timeout=30):

            raise RuntimeError(
                "❌ Qwen WebSocket connection timeout."
            )

    # =====================================================
    # ASYNCIO EVENT LOOP
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
            "🔌 Connecting to Qwen WebSocket...",
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
            "🟢 Qwen WebSocket connected.",
            flush=True
        )

    # =====================================================
    # SEND REQUEST
    # =====================================================

    async def _generate_async(
        self,
        payload,
        output_queue
    ):

        try:

            request_start = time.perf_counter()

            await self.websocket.send(
                json.dumps(payload)
            )

            print(
                "📤 Qwen WebSocket request sent",
                flush=True
            )

            first_token = True
            ttft = None

            while True:

                message = await self.websocket.recv()

                data = json.loads(message)

                # -----------------------------------------
                # ERROR
                # -----------------------------------------

                if "error" in data:

                    output_queue.put({
                        "error": data["error"]
                    })

                    break

                # -----------------------------------------
                # STREAMED CONTENT
                # -----------------------------------------

                if "content" in data:

                    if first_token:

                        ttft = (
                            time.perf_counter()
                            - request_start
                        )

                        print(
                            f"🚀 Qwen WS TTFT: "
                            f"{ttft:.3f}s",
                            flush=True
                        )

                        first_token = False

                    output_queue.put({
                        "content": data["content"]
                    })

                # -----------------------------------------
                # RESPONSE FINISHED
                # -----------------------------------------

                if data.get("done"):

                    total = (
                        time.perf_counter()
                        - request_start
                    )

                    print(
                        f"🏁 Qwen WS total: "
                        f"{total:.3f}s",
                        flush=True
                    )

                    output_queue.put({
                        "done": True
                    })

                    break

        except Exception as e:

            output_queue.put({
                "error": str(e)
            })

    # =====================================================
    # GENERATE
    # =====================================================

    def generate(
        self,
        payload
    ):

        # -------------------------------------------------
        # IMPORTANT:
        #
        # Only ONE request may use the WebSocket at a time.
        #
        # This prevents two Qwen requests from mixing
        # their streamed tokens.
        # -------------------------------------------------

        with self.request_lock:

            if not self.connected.is_set():

                raise RuntimeError(
                    "❌ Qwen WebSocket is not connected."
                )

            output_queue = __import__(
                "queue"
            ).Queue()

            future = asyncio.run_coroutine_threadsafe(

                self._generate_async(
                    payload,
                    output_queue
                ),

                self.loop
            )

            # ---------------------------------------------
            # YIELD STREAMED TOKENS
            # ---------------------------------------------

            while True:

                item = output_queue.get()

                if "error" in item:

                    raise RuntimeError(
                        item["error"]
                    )

                if "content" in item:

                    yield item["content"]

                if item.get("done"):

                    break

            # Make sure coroutine completed

            future.result()

    # =====================================================
    # CLOSE
    # =====================================================

    def close(self):

        if self.stopped.is_set():

            return

        self.stopped.set()

        print(
            "🔴 Closing Qwen WebSocket...",
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
            "🔴 Qwen WebSocket closed.",
            flush=True
        )