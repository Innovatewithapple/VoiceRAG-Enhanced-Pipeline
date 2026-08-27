import os
from dotenv import load_dotenv
from openai import OpenAI
from prompts.customer_support_agent import Customer_Support_Agent_Prompt
from prompts.Interruption_handling_prompt import INTERRUPTION_PROMPT
import requests
import json

load_dotenv()

QWEN_URL = "https://absentee-mulled-stadium.ngrok-free.dev"
MODEL_NAME = "Qwen3-30B-A3B-Q4_K_M.gguf"

qwen_client = None

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
    conversation_history,
    is_interruption=False
):

    # =========================================
    # SELECT PROMPT
    # =========================================

    if is_interruption:

        prompt = INTERRUPTION_PROMPT(
            interruption=query
        )

        max_tokens = 20
        temperature = 0.4
        top_p = 0.8

    else:

        prompt = Customer_Support_Agent_Prompt(
            query=query,
            context=context,
            source="terms and policy",
            conversation_history=conversation_history
        )

        max_tokens = 2048
        temperature = 0.7
        top_p = 0.8

        print("=" * 70)
        print("🔍 ACTUAL QWEN PROMPT DEBUG", flush=True)

        print(
            f"📦 Context type: {type(context)}",
            flush=True
        )

        print(
            f"📦 Context characters: {len(str(context))}",
            flush=True
        )

        print(
            f"📦 Prompt JSON characters: "
            f"{len(json.dumps(prompt))}",
            flush=True
        )

        for i, message in enumerate(prompt):

            content = message.get("content", "")

            print(
                f"Message {i} | "
                f"role={message.get('role')} | "
                f"characters={len(content)}",
                flush=True
            )

        print("=" * 70)

    # =========================================
    # QWEN
    # =========================================

    payload = {
    "model": MODEL_NAME,

    "messages": prompt,

    "temperature": temperature,

    "max_tokens": max_tokens,

    "top_p": top_p,

    "stream": True,

    "top_k": 20,

    "min_p": 0,

    "chat_template_kwargs": {
        "enable_thinking": False
    }
    }

    print(
    f"📝 Qwen prompt messages: {len(prompt)}",
    flush=True
    )

    print(
        f"📝 Conversation history messages: "
        f"{len(conversation_history)}",
        flush=True
    )

    response = qwen_client.generate(
        payload
    )

    full_response = ""

    # =========================================
    # INTERRUPTION
    # =========================================

    if is_interruption:

        for text in response:

            if not text:
                continue

            print(text,end="",flush=True)

            full_response += text

        # One tiny response → one TTS request
        if full_response.strip():

            tts_queue.put(
                full_response.strip()
            )

        tts_queue.put(None)

        return full_response.strip()

    # =========================================
    # NORMAL RESPONSE
    # =========================================

    buffer = ""

    for text in response:

        if not text:
            continue

        print(text,end="",flush=True)

        full_response += text
        buffer += text

        # =====================================
        # COMMA-FIRST TTS CHUNKING
        # =====================================

        while True:

            buffer = buffer.lstrip()

            if not buffer:
                break

            # ---------------------------------
            # Find comma and period
            # ---------------------------------

            comma_position = buffer.find(",")
            period_position = buffer.find(".")

            # ---------------------------------
            # Comma has priority
            # ---------------------------------

            if comma_position != -1:

                speech_text = (
                    buffer[:comma_position + 1]
                    .strip()
                )

                buffer = (
                    buffer[comma_position + 1:]
                    .lstrip()
                )

                if speech_text:

                    tts_queue.put(
                        speech_text
                    )

                continue

            # ---------------------------------
            # No comma → use period
            # ---------------------------------

            if period_position != -1:

                speech_text = (
                    buffer[:period_position + 1]
                    .strip()
                )

                buffer = (
                    buffer[period_position + 1:]
                    .lstrip()
                )

                if speech_text:

                    tts_queue.put(
                        speech_text
                    )

                continue

            # ---------------------------------
            # Nothing ready yet
            # ---------------------------------

            break

    # =========================================
    # FLUSH REMAINING TEXT
    # =========================================

    if buffer.strip():

        tts_queue.put(
            buffer.strip()
        )

    # =========================================
    # RESPONSE FINISHED
    # =========================================

    tts_queue.put(None)

    return full_response.strip()

# def Stream_LLM_To_TTS(
#     query,
#     context,
#     tts_queue,
#     conversation_history,
#     is_interruption=False
# ):

#     # =========================================
#     # SELECT PROMPT
#     # =========================================

#     if is_interruption:

#         prompt = INTERRUPTION_PROMPT(
#             interruption=query
#         )

#         max_tokens = 20
#         temperature = 0.4
#         top_p = 0.8

#     else:

#         prompt = Customer_Support_Agent_Prompt(
#             query=query,
#             context=context,
#             source="terms and policy",
#             conversation_history=conversation_history
#         )

#         max_tokens = 2048
#         temperature = 0.7
#         top_p = 0.8

#     # =========================================
#     # QWEN
#     # =========================================

#     response = client.chat.completions.create(
#         model=MODEL_NAME,
#         messages=prompt,
#         temperature=temperature,
#         max_tokens=max_tokens,
#         top_p=top_p,
#         stream=True,
#         extra_body={
#             "top_k": 20,
#             "min_p": 0,
#             "chat_template_kwargs": {
#                 "enable_thinking": False
#             }
#         }
#     )

#     full_response = ""

#     # =========================================
#     # INTERRUPTION
#     # =========================================

#     if is_interruption:

#         for chunk in response:

#             if not chunk.choices:
#                 continue

#             text = chunk.choices[0].delta.content

#             if not text:
#                 continue

#             print(
#                 text,
#                 end="",
#                 flush=True
#             )

#             full_response += text

#         # One tiny response → one TTS request
#         if full_response.strip():

#             tts_queue.put(
#                 full_response.strip()
#             )

#         tts_queue.put(None)

#         return full_response.strip()

#     # =========================================
#     # NORMAL RESPONSE
#     # =========================================

#     buffer = ""

#     for chunk in response:

#         if not chunk.choices:
#             continue

#         text = chunk.choices[0].delta.content

#         if not text:
#             continue

#         print(
#             text,
#             end="",
#             flush=True
#         )

#         full_response += text
#         buffer += text

#         # =====================================
#         # PERIOD-BASED TTS CHUNKING
#         # =====================================

#         while True:

#             buffer = buffer.lstrip()

#             if not buffer:
#                 break

#             period_position = buffer.find(".")

#             if period_position == -1:
#                 break

#             speech_text = (
#                 buffer[:period_position + 1]
#                 .strip()
#             )

#             buffer = (
#                 buffer[period_position + 1:]
#                 .lstrip()
#             )

#             if speech_text:

#                 tts_queue.put(
#                     speech_text
#                 )

#     # =========================================
#     # FLUSH REMAINING TEXT
#     # =========================================

#     if buffer.strip():

#         tts_queue.put(
#             buffer.strip()
#         )

#     # =========================================
#     # RESPONSE FINISHED
#     # =========================================

#     tts_queue.put(None)

#     return full_response.strip()

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