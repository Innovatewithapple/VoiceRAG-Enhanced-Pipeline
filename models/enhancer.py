from deepfilternet_rs import DeepFilterNetRealtime
import numpy as np
import librosa
import soundfile as sf

processor = None

INPUT_SR = 16000

def Initialize_Enhancer():

    global processor

    processor = DeepFilterNetRealtime(
        model_path=None,
        atten_lim=50.0,
        log_level="warn",
        compensate_delay=True,
        post_filter_beta=0.0,
    )


def Enhance_Audio(audio,output_path='/Users/himanshuvyas/VOICERAG/first.wav'):
    # ---convert audio into float32
    audio = np.asarray(audio, dtype=np.float32).flatten()
    MODEL_SR = processor.sample_rate
    # DeepFilterNet expects 48 kHz internally
    print("Input sample rate :", INPUT_SR)
    print("DeepFilterNet rate:", MODEL_SR)

    audio48K = librosa.resample(audio, orig_sr=INPUT_SR, target_sr=MODEL_SR).astype(
        np.float32
    )

    # Enhance
    enhanced_48K = processor.process_chunk(audio48K)

    # Flush remaining buffered audio
    #   tail = processor.finalize()

    #   enhanced_48K = np.concatenate([enhanced_48K,tail])

    # Resample back to 16khz for Whisper/VAD
    enhanced_16K = librosa.resample(
        enhanced_48K, orig_sr=MODEL_SR, target_sr=INPUT_SR
    ).astype(np.float32)
    sf.write(output_path,enhanced_16K,INPUT_SR)

    return enhanced_16K
