import time
import queue

from models.llm import Stream_LLM_To_TTS
import models.llm as llm

from models.qwen_websocket import QwenWebSocketClient


def test_llm():

    # =====================================================
    # START PERSISTENT QWEN WEBSOCKET
    # =====================================================

    print("🔌 Starting Qwen WebSocket...", flush=True)

    llm.qwen_client = QwenWebSocketClient()

    print(
        "🟢 Qwen WebSocket ready\n",
        flush=True
    )

    # =====================================================
    # TEST INPUT
    # =====================================================

    query = "Why can my application be cancelled?"

    context = """VisaFlow reserves the right to refuse or cancel an application in certain circumstances, including but not limited to the following:

i) The applicant provides incorrect, incomplete or misleading information.

ii) Required information or documents cannot be verified.

VisaFlow may also refuse or cancel an application if fraudulent, unauthorized or illegal activity is suspected.
"""

    conversation_history = []

    tts_queue = queue.Queue()

    # =====================================================
    # RUN STREAM_LLM_TO_TTS
    # =====================================================

    print(
        "🟢 Starting Stream_LLM_To_TTS test\n",
        flush=True
    )

    start = time.perf_counter()

    answer = Stream_LLM_To_TTS(
        query=query,
        context=context,
        tts_queue=tts_queue,
        conversation_history=conversation_history
    )

    total = time.perf_counter() - start

    # =====================================================
    # RESULTS
    # =====================================================

    print("\n")
    print("=" * 60)
    print("📊 STREAM_LLM_TO_TTS TEST")
    print("=" * 60)

    print(
        f"🤖 Function total: {total:.3f}s"
    )

    print(
        f"📝 Characters: {len(answer)}"
    )

    print("=" * 60)

    # =====================================================
    # IMPORTANT:
    # DO NOT CLOSE QWEN WEBSOCKET
    # =====================================================

    print(
        "🟢 Qwen WebSocket remains OPEN.",
        flush=True
    )


if __name__ == "__main__":

    test_llm()