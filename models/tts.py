from kokoro import KPipeline
import torch
import numpy as np

tts_pipeline = KPipeline(lang_code='a')

tts_pipeline.load_voice("af_bella")
tts_pipeline.load_voice("am_michael")

def Generate_Speech(text,voice):
    generator = tts_pipeline(text=text,voice=voice,speed=0.7)
    audio_segments = []

    for _,_,audio in generator:
        if torch.torch.is_tensor(audio):
            audio = audio.cpu().numpy()

        audio_segments.append(audio)

    final_audio = np.concatenate(audio_segments)
    return final_audio

