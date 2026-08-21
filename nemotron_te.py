import numpy as np
import torch

from threading import Thread

from transformers import (
    AutoProcessor,
    AutoModelForRNNT,
    TextIteratorStreamer,
)

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

model.eval()

# -----------------------------------------
# Streaming configuration
# -----------------------------------------

processor.set_num_lookahead_tokens(6)

print(
    "Streaming latency:",
    processor.streaming_latency_ms,
    "ms"
)

sampling_rate = (
    processor.feature_extractor.sampling_rate
)

# -----------------------------------------
# Load recorded audio
# -----------------------------------------

audio = load_audio(
    "/Users/himanshuvyas/VOICERAG/clean.wav",
    sampling_rate=sampling_rate,
)

print(
    "Audio:",
    len(audio) / sampling_rate,
    "seconds"
)

# -----------------------------------------
# FIRST CHUNK
# -----------------------------------------

first_chunk_inputs = processor(
    audio[
        :processor.num_samples_first_audio_chunk
    ],
    sampling_rate=sampling_rate,
    is_streaming=True,
    is_first_audio_chunk=True,
    return_tensors="pt",
)

first_chunk_inputs = first_chunk_inputs.to(
    model.device,
    dtype=model.dtype,
)

print(
    "First chunk:",
    processor.num_samples_first_audio_chunk,
    "samples"
)

# -----------------------------------------
# SUBSEQUENT CHUNKS
# -----------------------------------------

def input_features_generator():

    yield (
        first_chunk_inputs.input_features[
            :,
            :processor.num_mel_frames_first_audio_chunk,
            :
        ]
    )

    mel_frame_idx = (
        processor.num_mel_frames_first_audio_chunk
    )

    hop_length = (
        processor.feature_extractor.hop_length
    )

    n_fft = (
        processor.feature_extractor.n_fft
    )

    start_idx = (
        mel_frame_idx * hop_length
        - n_fft // 2
    )

    while True:

        end_idx = (
            start_idx
            + processor.num_samples_per_audio_chunk
        )

        if end_idx >= len(audio):
            break

        inputs = processor(
            audio[start_idx:end_idx],
            sampling_rate=sampling_rate,
            is_streaming=True,
            is_first_audio_chunk=False,
            return_tensors="pt",
        )

        inputs = inputs.to(
            model.device,
            dtype=model.dtype,
        )

        yield inputs.input_features

        mel_frame_idx += (
            processor.num_mel_frames_per_audio_chunk
        )

        start_idx = (
            mel_frame_idx * hop_length
            - n_fft // 2
        )


# -----------------------------------------
# STREAMER
# -----------------------------------------

streamer = TextIteratorStreamer(
    processor.tokenizer,
    skip_special_tokens=True,
)

generate_kwargs = {
    **first_chunk_inputs,
    "input_features": input_features_generator(),
    "streamer": streamer,
    "max_new_tokens": 128,
}

print("\n🚀 Starting streaming inference...")
print("-" * 60)

thread = Thread(
    target=model.generate,
    kwargs=generate_kwargs,
)

thread.start()

for text_chunk in streamer:

    print(
        text_chunk,
        end="",
        flush=True,
    )

thread.join()

print("\n")
print("-" * 60)
print("✅ Streaming test finished")