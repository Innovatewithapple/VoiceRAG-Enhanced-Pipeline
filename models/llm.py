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


def Stream_LLM_To_TTS(query, context, tts_queue):

    prompt = Customer_Support_Agent_Prompt(
        query=query,
        context=context,
        source="terms and policy"
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=prompt,
        temperature=0.2,
        max_tokens=2048,
        stream=True
    )

    full_response = ""
    buffer = ""

    MAX_WORDS = 15

    for chunk in response:

        if not chunk.choices:
            continue

        text = chunk.choices[0].delta.content

        if not text:
            continue

        # Show LLM response live
        print(text,end="",flush=True)

        full_response += text
        buffer += text

        # -----------------------------------------
        # Check whether enough text is ready
        # -----------------------------------------

        while True:

            # -----------------------------------------
            # Check punctuation
            # -----------------------------------------

            positions = [
                buffer.find(","),
                buffer.find("."),
                buffer.find("?"),
                buffer.find("!")
            ]

            positions = [
                p for p in positions
                if p != -1
            ]

            # -----------------------------------------
            # Punctuation found BEFORE 12 words
            # -----------------------------------------

            if positions:

                end = min(positions)

                speech_text = (buffer[:end + 1].strip())

                buffer = (buffer[end + 1:].strip())

                if speech_text:
                    tts_queue.put(speech_text)

                continue

            # -----------------------------------------
            # Check word limit FIRST
            # -----------------------------------------

            words = buffer.split()

            if len(words) >= MAX_WORDS:
                speech_text = " ".join(words[:MAX_WORDS])
                buffer = " ".join(words[MAX_WORDS:])
                tts_queue.put(speech_text)

                continue            

            # -----------------------------------------
            # Nothing ready yet
            # -----------------------------------------

            break

    # ---------------------------------------------
    # Flush whatever remains
    # ---------------------------------------------

    if buffer.strip():
        tts_queue.put(buffer.strip())

    # Tell TTS worker that this LLM response is finished
    tts_queue.put(None)
    return full_response.strip()