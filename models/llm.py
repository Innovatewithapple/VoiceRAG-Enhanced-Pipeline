import os
from dotenv import load_dotenv
from openai import OpenAI
from prompts.customer_support_agent import Customer_Support_Agent_Prompt
import requests

load_dotenv()

# client = OpenAI(
#     base_url="https://integrate.api.nvidia.com/v1",
#     api_key=os.getenv("NVIDIA_API_KEY")
# )

# MODEL_NAME = 'nvidia/nemotron-3-super-120b-a12b' #"nvidia/ising-calibration-1.5-31b"

client = OpenAI(
    base_url="https://absentee-mulled-stadium.ngrok-free.dev/v1",
    api_key="anything"
)
QWEN_URL = "https://absentee-mulled-stadium.ngrok-free.dev"
MODEL_NAME = "Qwen3-30B-A3B-Q4_K_M.gguf"

def Generate_LLM_Response(query,context):
    prompt = Customer_Support_Agent_Prompt(query=query,context=context,source="terms and policy")
    response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=prompt,
    temperature=0.7,
    top_p=0.8,
    max_tokens=2048,
    stream=False,
    extra_body={
        "top_k": 20,
        "min_p": 0,
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    }
    )

    # full_response = ""
    for chunk in response:
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta.content

        if delta:
            yield delta
            # print(delta,end="",flush=True)
            # full_response += delta
    # print()
    # return full_response.strip()
MAX_CONTEXT = 16384
MAX_OUTPUT = 2048
MAX_INPUT = MAX_CONTEXT - MAX_OUTPUT   # 14336


def count_qwen_tokens(messages):
    response = requests.post(
        f"{QWEN_URL}/apply-template",
        json={"messages": messages},
        timeout=30
    )
    response.raise_for_status()

    formatted_prompt = response.json()["prompt"]

    response = requests.post(
        f"{QWEN_URL}/tokenize",
        json={"content": formatted_prompt},
        timeout=30
    )
    response.raise_for_status()

    return len(response.json()["tokens"])

def Stream_LLM_To_TTS(
    query,
    context,
    tts_queue,
    conversation_history
):

    prompt = Customer_Support_Agent_Prompt(
        query=query,
        context=context,
        source="terms and policy",
        conversation_history=conversation_history
    )

    print("=" * 50)
    print("Conversation_History: ")
    print(conversation_history)
    print("=" * 50)

    input_tokens = count_qwen_tokens(prompt)

    print(
        f"🧮 Input tokens: "
        f"{input_tokens}/{MAX_INPUT}"
    )

    if input_tokens > MAX_INPUT:

        print(
            f"⚠️ Context too long: "
            f"{input_tokens} input tokens > "
            f"{MAX_INPUT} available."
        )

        tts_queue.put(None)

        return ""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=prompt,
        temperature=0.7,
        max_tokens=2048,
        top_p=0.8,
        stream=True,
        extra_body={
            "top_k": 20,
            "min_p": 0,
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        }
    )

    full_response = ""
    buffer = ""

    # =========================================
    # STREAM LLM
    # =========================================

    for chunk in response:

        if not chunk.choices:
            continue

        text = chunk.choices[0].delta.content

        if not text:
            continue

        # -----------------------------------------
        # Show LLM response live
        # -----------------------------------------

        print(
            text,
            end="",
            flush=True
        )

        full_response += text
        buffer += text

        # =========================================
        # CHECK FOR PERIOD
        # =========================================

        while True:

            buffer = buffer.lstrip()

            if not buffer:
                break

            # -----------------------------------------
            # Find ONLY "."
            # -----------------------------------------

            period_position = buffer.find(".")

            if period_position == -1:
                break

            # -----------------------------------------
            # Everything through "." becomes
            # one TTS chunk
            # -----------------------------------------

            speech_text = (
                buffer[:period_position + 1]
                .strip()
            )

            # -----------------------------------------
            # Remove spoken text from buffer
            # -----------------------------------------

            buffer = (
                buffer[period_position + 1:]
                .lstrip()
            )

            # -----------------------------------------
            # Send to TTS
            # -----------------------------------------

            if speech_text:

                tts_queue.put(
                    speech_text
                )

    # =============================================
    # FLUSH REMAINING TEXT
    # =============================================

    if buffer.strip():

        tts_queue.put(
            buffer.strip()
        )

    # =============================================
    # TELL TTS WORKER RESPONSE IS FINISHED
    # =============================================

    tts_queue.put(None)

    return full_response.strip()

# def Stream_LLM_To_TTS(query, context, tts_queue,conversation_history):

#     prompt = Customer_Support_Agent_Prompt(
#         query=query,
#         context=context,
#         source="terms and policy",
#         conversation_history=conversation_history
#     )

#     print("="*50)
#     print("Conversation_History: ")
#     print(conversation_history)
#     print("="*50)

#     input_tokens = count_qwen_tokens(prompt)

#     print(f"🧮 Input tokens: {input_tokens}/{MAX_INPUT}")
#     if input_tokens > MAX_INPUT:
#         print(
#             f"⚠️ Context too long: "
#             f"{input_tokens} input tokens > {MAX_INPUT} available."
#         )
#         tts_queue.put(None)

#         return ""

#     response = client.chat.completions.create(
#         model=MODEL_NAME,
#         messages=prompt,
#         temperature=0.2,
#         max_tokens=2048,
#         stream=True
#     )

#     full_response = ""
#     buffer = ""

#     # =========================================
#     # CHUNKING SETTINGS
#     # =========================================

#     MAX_WORDS = 15

#     # Strong sentence punctuation:
#     # . ? !
#     MIN_SENTENCE_WORDS = 4

#     # Weaker punctuation:
#     # ,
#     MIN_COMMA_WORDS = 8

#     # =========================================
#     # STREAM LLM
#     # =========================================

#     for chunk in response:

#         if not chunk.choices:
#             continue

#         text = chunk.choices[0].delta.content

#         if not text:
#             continue

#         # -----------------------------------------
#         # Show LLM response live
#         # -----------------------------------------

#         print(
#             text,
#             end="",
#             flush=True
#         )

#         full_response += text
#         buffer += text

#         # =========================================
#         # PROCESS BUFFER
#         # =========================================

#         while True:

#             buffer = buffer.lstrip()

#             if not buffer:
#                 break

#             # =====================================
#             # FIND SENTENCE PUNCTUATION
#             # =====================================

#             sentence_positions = [
#                 buffer.find("."),
#                 buffer.find("?"),
#                 buffer.find("!")
#             ]

#             sentence_positions = [
#                 p for p in sentence_positions
#                 if p != -1
#             ]

#             # =====================================
#             # SENTENCE BOUNDARY
#             # =====================================

#             if sentence_positions:

#                 end = min(sentence_positions)

#                 candidate = (
#                     buffer[:end + 1]
#                     .strip()
#                 )

#                 candidate_words = candidate.split()

#                 # ---------------------------------
#                 # Only split if sentence is
#                 # reasonably large.
#                 # ---------------------------------

#                 if len(candidate_words) >= MIN_SENTENCE_WORDS:

#                     tts_queue.put(candidate)

#                     buffer = (
#                         buffer[end + 1:]
#                         .lstrip()
#                     )

#                     continue

#             # =====================================
#             # FIND COMMA
#             # =====================================

#             comma_position = buffer.find(",")

#             if comma_position != -1:

#                 candidate = (
#                     buffer[:comma_position + 1]
#                     .strip()
#                 )

#                 candidate_words = candidate.split()

#                 # ---------------------------------
#                 # Comma is a weaker boundary.
#                 # Only use it when we already
#                 # have enough words.
#                 # ---------------------------------

#                 if len(candidate_words) >= MIN_COMMA_WORDS:

#                     tts_queue.put(candidate)

#                     buffer = (
#                         buffer[comma_position + 1:]
#                         .lstrip()
#                     )

#                     continue

#             # =====================================
#             # MAX WORD LIMIT
#             # =====================================

#             words = buffer.split()

#             if len(words) >= MAX_WORDS:

#                 # ---------------------------------
#                 # Take first 15 words.
#                 # ---------------------------------

#                 speech_words = words[:MAX_WORDS]

#                 speech_text = " ".join(
#                     speech_words
#                 )

#                 # ---------------------------------
#                 # Remove those words from buffer.
#                 # ---------------------------------

#                 remaining_words = words[MAX_WORDS:]

#                 buffer = " ".join(
#                     remaining_words
#                 )

#                 # ---------------------------------
#                 # If punctuation belongs directly
#                 # to the 15th word, keep it with
#                 # the current chunk.
#                 #
#                 # Example:
#                 #
#                 # "application, the..."
#                 #
#                 # becomes:
#                 #
#                 # "application,"
#                 #
#                 # rather than leaving "," behind.
#                 # ---------------------------------

#                 if remaining_words:

#                     # Nothing else required here because
#                     # split() keeps punctuation attached
#                     # to its word.

#                     pass

#                 if speech_text:

#                     tts_queue.put(
#                         speech_text
#                     )

#                 continue

#             # =====================================
#             # NOTHING READY YET
#             # =====================================

#             break

#     # =============================================
#     # FLUSH REMAINING TEXT
#     # =============================================

#     if buffer.strip():

#         tts_queue.put(
#             buffer.strip()
#         )

#     # =============================================
#     # TELL TTS WORKER THIS RESPONSE IS FINISHED
#     # =============================================

#     tts_queue.put(None)

#     return full_response.strip()