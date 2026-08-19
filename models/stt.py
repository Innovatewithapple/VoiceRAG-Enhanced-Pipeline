from transformers import pipeline

stt = pipeline("automatic-speech-recognition",
               model="openai/whisper-small",
               device="mps")


def Transcribe(audio):
    result = stt(audio,generate_kwargs={
            "language": "English",
            "task": "transcribe"
        })

    return result['text'].strip() 