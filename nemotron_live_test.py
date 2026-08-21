import queue
import threading
import numpy as np
import sounddevice as sd
import torch

from transformers import (
    AutoProcessor,
    AutoModelForRNNT,
    TextIteratorStreamer,
)


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_ID = "nvidia/nemotron-speech-streaming-en-0.6b"

SAMPLE_RATE = 16000
CHANNELS = 1

# NVIDIA streaming configuration
LOOKAHEAD_TOKENS = 6


# =========================================================
# LOAD MODEL
# =========================================================

print("Loading processor...")

processor = AutoProcessor.from_pretrained(
    MODEL_ID
)

print("Loading Nemotron...")

model = AutoModelForRNNT.from_pretrained(
    MODEL_ID
)

model.eval()

print("✅ Nemotron loaded")


# =========================================================
# STREAMING CONFIGURATION
# =========================================================

processor.set_num_lookahead_tokens(
    LOOKAHEAD_TOKENS
)

print(
    f"Streaming latency: "
    f"{processor.streaming_latency_ms} ms"
)

print(
    f"First chunk: "
    f"{processor.num_samples_first_audio_chunk} samples"
)

print(
    f"Subsequent chunk: "
    f"{processor.num_samples_per_audio_chunk} samples"
)


# =========================================================
# AUDIO QUEUE
# =========================================================

audio_queue = queue.Queue()


# =========================================================
# MICROPHONE CALLBACK
# =========================================================

def audio_callback(
    indata,
    frames,
    time,
    status
):

    if status:
        print("STATUS:", status)

    audio = indata[:, 0].copy()

    audio_queue.put(audio)


# =========================================================
# GET REQUIRED AUDIO FROM QUEUE
# =========================================================

def get_audio(required_samples):

    chunks = []
    total_samples = 0

    while total_samples < required_samples:

        audio = audio_queue.get()

        chunks.append(audio)

        total_samples += len(audio)

    audio = np.concatenate(chunks)

    return audio[:required_samples]


# =========================================================
# NEMOTRON STREAMING
# =========================================================

def run_nemotron():

    print("\n🟢 Nemotron worker started")

    # -----------------------------------------
    # FIRST CHUNK
    # -----------------------------------------

    first_samples = (
        processor.num_samples_first_audio_chunk
    )

    print(
        f"⏳ Waiting for "
        f"{first_samples} samples..."
    )

    first_audio = get_audio(
        first_samples
    )

    print(
        f"🎤 First chunk received: "
        f"{len(first_audio)} samples"
    )

    # -----------------------------------------
    # PROCESS FIRST CHUNK
    # -----------------------------------------

    first_inputs = processor(
        first_audio,
        sampling_rate=SAMPLE_RATE,
        is_streaming=True,
        is_first_audio_chunk=True,
        return_tensors="pt",
    )

    first_inputs = first_inputs.to(
        model.device,
        dtype=model.dtype
    )

    print("✅ First chunk processed")

    # -----------------------------------------
    # SUBSEQUENT AUDIO GENERATOR
    # -----------------------------------------

    def input_features_generator():

        # IMPORTANT:
        # First chunk must be sliced to the exact
        # number of mel frames required by Nemotron.

        yield (
            first_inputs.input_features[
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

        # -----------------------------------------
        # Continue with live microphone chunks
        # -----------------------------------------

        while True:

            samples = (
                processor.num_samples_per_audio_chunk
            )

            audio = get_audio(samples)

            inputs = processor(
                audio,
                sampling_rate=SAMPLE_RATE,
                is_streaming=True,
                is_first_audio_chunk=False,
                return_tensors="pt",
            )

            inputs = inputs.to(
                model.device,
                dtype=model.dtype
            )

            yield inputs.input_features

            mel_frame_idx += (
                processor.num_mel_frames_per_audio_chunk
            )


    # -----------------------------------------
    # STREAMER
    # -----------------------------------------

    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_special_tokens=True
    )


    # -----------------------------------------
    # GENERATION
    # -----------------------------------------

    generate_kwargs = {
        **first_inputs,
        "input_features": input_features_generator(),
        "streamer": streamer,
    }

    print("🚀 Starting Nemotron...")

    generation_thread = threading.Thread(
        target=model.generate,
        kwargs=generate_kwargs,
        daemon=True
    )

    generation_thread.start()

    print("🟢 Listening / transcribing...")
    print("-" * 60)

    for text in streamer:

        print(
            text,
            end="",
            flush=True
        )

# =========================================================
# START
# =========================================================

worker = threading.Thread(
    target=run_nemotron,
    daemon=True
)

worker.start()


# =========================================================
# MICROPHONE
# =========================================================

print("\n🎤 Starting microphone...")
print("Speak now. Press Ctrl+C to stop.\n")


try:

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=512,
        callback=audio_callback,
    ):

        while True:
            sd.sleep(1000)


except KeyboardInterrupt:

    print("\n🛑 Microphone stopped.")