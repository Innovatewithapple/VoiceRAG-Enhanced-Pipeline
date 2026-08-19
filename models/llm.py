import os
from dotenv import load_dotenv
from openai import OpenAI
from prompts.customer_support_agent import Customer_Support_Agent_Prompt

load_dotenv()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

MODEL_NAME = "nvidia/ising-calibration-1.5-31b"

def Generate_LLM_Response(query,context):
    prompt = Customer_Support_Agent_Prompt(query=query,context=context,source="terms and policy")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=prompt,
        temperature=0.2,
        max_tokens=2048,
        stream=False
    )

    return response.choices[0].message.content.strip()