import time

from models.qwen_websocket import QwenWebSocketClient
from prompts.customer_support_agent import Customer_Support_Agent_Prompt


def test_client():

    client = QwenWebSocketClient()

    print("\n🟢 Client ready\n")

    queries = [
        "Why can my application be cancelled?",
        "Why can my application be cancelled?",
        "Why can my application be cancelled?",
        "Why can my application be cancelled?",
        "Why can my application be cancelled?"
    ]

    context = """VisaFlow reserves the right to refuse or cancel an application in certain circumstances, including but not limited to the following:

i) The applicant provides incorrect, incomplete or misleading information.

ii) Required information or documents cannot be verified.

VisaFlow may also refuse or cancel an application if fraudulent, unauthorized or illegal activity is suspected.
"""

    for i, query in enumerate(queries):

        prompt = Customer_Support_Agent_Prompt(
            query=query,
            context=context,
            source="terms and policy",
            conversation_history=[]
        )

        payload = {
            "model": "Qwen3-30B-A3B-Q4_K_M.gguf",

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

        print("=" * 60)
        print(f"Query {i + 1}")

        start = time.perf_counter()

        first_token = True
        ttft = None
        full_response = ""

        for text in client.generate(payload):

            if not text:
                continue

            if first_token:

                ttft = (
                    time.perf_counter()
                    - start
                )

                first_token = False

            full_response += text

        total = (
            time.perf_counter()
            - start
        )

        print(
            f"🚀 Client TTFT: {ttft:.3f}s"
        )

        print(
            f"🤖 Client total: {total:.3f}s"
        )

        print(
            f"📝 Characters: {len(full_response)}"
        )

    print("\n🔵 Test finished.")

    # DO NOT CLOSE CLIENT HERE


if __name__ == "__main__":

    test_client()