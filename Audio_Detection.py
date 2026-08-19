import torch
import numpy as np
from models.enhancer import Enhance_Audio
from models.stt import Transcribe
from models.vad import vad_iterator


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
#                       Whisper

SAMPLE_RATE=16000
CHANNELS=1
CHUNKSIZE= 512

speech_audio=[]
is_speaking=False

def Detect_Speech_And_Process_Audio(audio):
    global speech_audio,is_speaking

    audio_tensor = torch.from_numpy(audio).float()

    #Ask VAD whether speech state changed
    speech_event = vad_iterator(audio_tensor)

    if speech_event and "start" in speech_event:
        is_speaking=True

        print("🎤 Speech started")
        print(speech_event)

        speech_audio=[]
        speech_audio.append(audio)
        return None

    elif is_speaking:
       # print("🔵 still speaking.....")
        speech_audio.append(audio)

    if speech_event and "end" in speech_event:
        is_speaking=False

        print("🛑 Speech ended")
        utterance = np.concatenate(speech_audio)

        speech_audio = []
        print(
            f"Complete utterance: "
            f"{len(utterance)} samples "
            f"({len(utterance) / SAMPLE_RATE:.2f} seconds)"
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