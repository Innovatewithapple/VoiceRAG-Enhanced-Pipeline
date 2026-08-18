from Audio_Detection import Detect_Speech_And_Process_Audio,SAMPLE_RATE,CHANNELS,CHUNKSIZE
import sounddevice as sd

def audio_callback(indata, frames, time, status):

    if status:
        print("STATUS:", status)

    # Get microphone audio
    audio = indata[:, 0].copy()

    # Give chunk to VAD + buffer
    utterance = Detect_Speech_And_Process_Audio(audio)

    # --------------------------------------
    # COMPLETE UTTERANCE
    # --------------------------------------

    if utterance is not None:

        print("✅ Complete audio ready")
        print("="*50)
        print("\n")
        print("🎤 Listening.......")

        # Later:
        #
        # enhanced_audio = enhance(utterance)
        #
        # text = whisper(enhanced_audio)
        #
        # RAG...


# ==========================================
# START MICROPHONE
# ==========================================

try:

    with sd.InputStream(samplerate=SAMPLE_RATE,channels=CHANNELS,dtype="float32",blocksize=CHUNKSIZE,callback=audio_callback):
        print("🎤 Listening.......")
        while True:
            sd.sleep(1000)

except KeyboardInterrupt:
    print("\n🛑 Microphone stopped.")