import os
from dotenv import load_dotenv
from openai import OpenAI
from prompts.customer_support_agent import Customer_Support_Agent_Prompt

load_dotenv()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

MODEL_NAME = 'nvidia/nemotron-3-super-120b-a12b' #"nvidia/ising-calibration-1.5-31b"

def Generate_LLM_Response(query,context):
    prompt = Customer_Support_Agent_Prompt(query=query,context=context,source="terms and policy")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=prompt,
        temperature=0.2,
        max_tokens=2048,
        stream=True
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


def Stream_LLM_To_TTS(query, context, tts_queue,conversation_history):

    prompt = Customer_Support_Agent_Prompt(
        query=query,
        context=context,
        source="terms and policy",
        conversation_history=conversation_history
    )

    print("="*50)
    print("Conversation_History: ")
    print(conversation_history)
    print("="*50)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=prompt,
        temperature=0.2,
        max_tokens=2048,
        stream=True
    )

    full_response = ""
    buffer = ""

    # =========================================
    # CHUNKING SETTINGS
    # =========================================

    MAX_WORDS = 15

    # Strong sentence punctuation:
    # . ? !
    MIN_SENTENCE_WORDS = 4

    # Weaker punctuation:
    # ,
    MIN_COMMA_WORDS = 8

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
        # PROCESS BUFFER
        # =========================================

        while True:

            buffer = buffer.lstrip()

            if not buffer:
                break

            # =====================================
            # FIND SENTENCE PUNCTUATION
            # =====================================

            sentence_positions = [
                buffer.find("."),
                buffer.find("?"),
                buffer.find("!")
            ]

            sentence_positions = [
                p for p in sentence_positions
                if p != -1
            ]

            # =====================================
            # SENTENCE BOUNDARY
            # =====================================

            if sentence_positions:

                end = min(sentence_positions)

                candidate = (
                    buffer[:end + 1]
                    .strip()
                )

                candidate_words = candidate.split()

                # ---------------------------------
                # Only split if sentence is
                # reasonably large.
                # ---------------------------------

                if len(candidate_words) >= MIN_SENTENCE_WORDS:

                    tts_queue.put(candidate)

                    buffer = (
                        buffer[end + 1:]
                        .lstrip()
                    )

                    continue

            # =====================================
            # FIND COMMA
            # =====================================

            comma_position = buffer.find(",")

            if comma_position != -1:

                candidate = (
                    buffer[:comma_position + 1]
                    .strip()
                )

                candidate_words = candidate.split()

                # ---------------------------------
                # Comma is a weaker boundary.
                # Only use it when we already
                # have enough words.
                # ---------------------------------

                if len(candidate_words) >= MIN_COMMA_WORDS:

                    tts_queue.put(candidate)

                    buffer = (
                        buffer[comma_position + 1:]
                        .lstrip()
                    )

                    continue

            # =====================================
            # MAX WORD LIMIT
            # =====================================

            words = buffer.split()

            if len(words) >= MAX_WORDS:

                # ---------------------------------
                # Take first 15 words.
                # ---------------------------------

                speech_words = words[:MAX_WORDS]

                speech_text = " ".join(
                    speech_words
                )

                # ---------------------------------
                # Remove those words from buffer.
                # ---------------------------------

                remaining_words = words[MAX_WORDS:]

                buffer = " ".join(
                    remaining_words
                )

                # ---------------------------------
                # If punctuation belongs directly
                # to the 15th word, keep it with
                # the current chunk.
                #
                # Example:
                #
                # "application, the..."
                #
                # becomes:
                #
                # "application,"
                #
                # rather than leaving "," behind.
                # ---------------------------------

                if remaining_words:

                    # Nothing else required here because
                    # split() keeps punctuation attached
                    # to its word.

                    pass

                if speech_text:

                    tts_queue.put(
                        speech_text
                    )

                continue

            # =====================================
            # NOTHING READY YET
            # =====================================

            break

    # =============================================
    # FLUSH REMAINING TEXT
    # =============================================

    if buffer.strip():

        tts_queue.put(
            buffer.strip()
        )

    # =============================================
    # TELL TTS WORKER THIS RESPONSE IS FINISHED
    # =============================================

    tts_queue.put(None)

    return full_response.strip()