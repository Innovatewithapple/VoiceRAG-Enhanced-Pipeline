import asyncio
import json
import time
import websockets
from prompts.customer_support_agent import Customer_Support_Agent_Prompt

QWEN_WS_URL = (
    "wss://absentee-mulled-stadium.ngrok-free.dev/ws"
)


async def test_qwen():

    async with websockets.connect(
        QWEN_WS_URL,
        ping_interval=20,
        ping_timeout=60,
        max_size=None
    ) as ws:

        print("🟢 Connected to Qwen WebSocket\n")

        queries = [
            "Why can my application be cancelled?",
            "Why can my application be cancelled?",
            "Why can my application be cancelled?",
            "Why can my application be cancelled?",
            "Why can my application be cancelled?"
        ]

        for i, query in enumerate(queries):

            # =========================================
            # YOUR REAL PROMPT
            # =========================================

            prompt = Customer_Support_Agent_Prompt(
                query=query,
                context="""VisaFlow reserves the right to refuse or cancel an application in certain circumstances, including but not limited to the following:

i) The applicant provides incorrect, incomplete or misleading information.

ii) Required information or documents cannot be verified.

VisaFlow may also refuse or cancel an application if fraudulent, unauthorized or illegal activity is suspected.
""",
                source="terms and policy",
                conversation_history=[]
            )

            payload = {
                "model": 'Qwen3-30B-A3B-Q4_K_M.gguf',

                "messages": prompt,

                "temperature": 0.7,

                "max_tokens": 2048,

                "top_p": 0.8,

                "stream": True,

                "top_k": 20,

                "min_p": 0,

                "chat_template_kwargs": {
                    "enable_thinking": False
                }
            }

            # =========================================
            # SEND
            # =========================================

            start = time.perf_counter()

            await ws.send(
                json.dumps(payload)
            )

            first_token = True
            ttft = None
            full_response = ""

            # =========================================
            # RECEIVE
            # =========================================

            while True:

                message = await ws.recv()

                data = json.loads(message)

                if "error" in data:

                    print(
                        "❌ Error:",
                        data["error"]
                    )

                    break

                if "content" in data:

                    if first_token:

                        ttft = (
                            time.perf_counter()
                            - start
                        )

                        first_token = False

                    text = data['content']
                    print(
                        text,
                        end="",
                        flush=True
                    )

                    full_response += data["content"]

                if data.get("done"):

                    total = (
                        time.perf_counter()
                        - start
                    )
                    print()

                    break

            print("=" * 60)
            print(f"Query {i + 1}: {query}")
            print(
                f"🚀 TTFT: {ttft:.3f}s"
            )
            print(
                f"🤖 Total: {total:.3f}s"
            )

        print("\n🔵 All queries finished.")
        print("🔵 WebSocket connection will now close.")


asyncio.run(test_qwen())