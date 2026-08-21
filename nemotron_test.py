import torch

from transformers import AutoProcessor, AutoModelForRNNT
from transformers.audio_utils import load_audio


MODEL_ID = "nvidia/nemotron-speech-streaming-en-0.6b"


print("Loading processor...")

processor = AutoProcessor.from_pretrained(
    MODEL_ID
)

print("Loading model...")

model = AutoModelForRNNT.from_pretrained(
    MODEL_ID
)

print("✅ Model loaded")

print(
    "Sample rate:",
    processor.feature_extractor.sampling_rate
)


audio = load_audio(
    "/Users/himanshuvyas/VOICERAG/noisy.mp3",
    sampling_rate=processor.feature_extractor.sampling_rate
)

print(
    "Audio duration:",
    len(audio) /
    processor.feature_extractor.sampling_rate,
    "seconds"
)

inputs = processor(
    audio,
    sampling_rate=processor.feature_extractor.sampling_rate
)

inputs = inputs.to(
    model.device,
    dtype=model.dtype
)

output = model.generate(
    **inputs,
    return_dict_in_generate=True
)

text = processor.decode(
    output.sequences,
    skip_special_tokens=True
)

print("📝 Transcription:")
print(text)