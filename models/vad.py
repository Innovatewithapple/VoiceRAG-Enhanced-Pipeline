from silero_vad import load_silero_vad,VADIterator
vad_model = load_silero_vad()

vad_iterator = VADIterator(model=vad_model,
                           threshold=0.5,
                           sampling_rate=16000,
                           min_silence_duration_ms=1600,
                           speech_pad_ms=30)