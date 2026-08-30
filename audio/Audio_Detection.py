import torch
import numpy as np
import audio.audio_state as audio_state
from models.enhancer import Enhance_Audio
from models.stt import Transcribe
from models.vad import vad_iterator
from audio.audio_state import (
    tts_speaking_event,
    tts_interrupt_event
)


# 🎤 Microphone
#       ↓
#   small audio chunk
#       ↓
#     Silero VAD (need to check intra turn pause where user take just a pause to think)
#       │
#       ├── NO SPEECH → discard / don't send to STT
#       │
#       └── SPEECH → start collecting audio
#                          ↓
#                     DeepFilterNet
#                          ↓
#                     speech ends
#                          ↓
#                       nemotron

SAMPLE_RATE=16000
CHANNELS=1
CHUNKSIZE= 512

speech_audio = []
is_speaking = False


def Detect_Speech_And_Process_Audio(audio):

    global speech_audio, is_speaking

    audio_tensor = torch.from_numpy(audio).float()

    # =====================================================
    # ASK VAD WHETHER SPEECH STATE CHANGED
    # =====================================================

    speech_event = vad_iterator(audio_tensor)

    # =====================================================
    # SPEECH START
    # =====================================================

    if speech_event and "start" in speech_event:

        is_speaking = True

        print(
            "🎤 Speech started",
            flush=True
        )

        print(
            speech_event,
            flush=True
        )

        # =================================================
        # INTERRUPT CURRENT TTS
        # =================================================

        if tts_speaking_event.is_set():

            print(
                "🛑 USER INTERRUPTED → STOPPING TTS",
                flush=True
            )

            # Stop current playback
            tts_interrupt_event.set()

            # Invalidate ALL audio belonging to the old response
            audio_state.tts_generation_id += 1

            print(
                f"🗑️ Old TTS generation invalidated → "
                f"{audio_state.tts_generation_id}",
                flush=True
            )

        else:

            print(
                "ℹ️ TTS is not speaking",
                flush=True
            )

        # =================================================
        # START COLLECTING USER SPEECH
        # =================================================

        speech_audio = []

        speech_audio.append(audio)

        return None

    # =====================================================
    # CONTINUE COLLECTING SPEECH
    # =====================================================

    elif is_speaking:

        speech_audio.append(audio)

    # =====================================================
    # SPEECH END
    # =====================================================

    if speech_event and "end" in speech_event:

        is_speaking = False

        print(
            "🛑 Speech ended",
            flush=True
        )

        utterance = np.concatenate(
            speech_audio
        )

        speech_audio = []

        print(
            f"Complete utterance: "
            f"{len(utterance)} samples "
            f"({len(utterance) / SAMPLE_RATE:.2f} seconds)",
            flush=True
        )

        return utterance

    return None







#Error case that was happening but then we solved it:

# That was a bad diagnostic setup because VADIterator maintains internal model state. We were effectively doing:

# audio chunk
#    ↓
# vad_model()
#    ↓
# same audio chunk
#    ↓
# VADIterator
#    ↓
# vad_model()

# So we were disturbing the state that VADIterator was supposed to maintain.