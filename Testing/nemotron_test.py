import asyncio
import time
import json
import websockets


COLAB_WS_URL = (
    "wss://plethora-registry-shrine.ngrok-free.dev/ws"
)


async def test_websocket():

    async with websockets.connect(
        COLAB_WS_URL,
        ping_interval=20,
        ping_timeout=20
    ) as websocket:

        print("🟢 WebSocket connected\n")

        queries = [
            "What is the refund policy?",
            "Hi, i m mihir and i have a query regarding cancellation and i want to know Why can my application be cancelled?",
            "How long does the application take?",
            "What documents do I need?",
            "Can I cancel my application?"
        ]

        for i, query in enumerate(queries):

            start = time.perf_counter()

            await websocket.send(
                json.dumps({
                    "query": query
                })
            )

            response = await websocket.recv()

            elapsed = (
                time.perf_counter()
                - start
            )

            data = json.loads(response)

            print("=" * 60)
            print("=" * 60)
            print(f"Query {i + 1}: {query}")

            print(
                f"🌐 WebSocket total: "
                f"{elapsed:.3f}s"
            )

            print(
                "📦 Server response:"
            )

            print(data)
            # print(f"Query {i + 1}: {query}")

            # print(
            #     f"🌐 WebSocket total: "
            #     f"{elapsed:.3f}s"
            # )

            # print(
            #     f"📚 Colab retrieval: "
            #     f"{data['retrieval_time']:.3f}s"
            # )

            # print(
            #     f"🔄 Colab reranking: "
            #     f"{data['reranking_time']:.3f}s"
            # )

            # print(
            #     f"⚡ Colab processing: "
            #     f"{data['total_time']:.3f}s"
            # )


asyncio.run(test_websocket())