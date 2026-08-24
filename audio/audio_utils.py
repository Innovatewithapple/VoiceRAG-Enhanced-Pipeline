import numpy as np

def get_silence_duration(audio, threshold=0.005):

    audio = np.asarray(audio)

    if audio.ndim > 1:
        audio = audio[:, 0]

    active = np.where(
        np.abs(audio) > threshold
    )[0]

    if len(active) == 0:
        return (
            len(audio) / 24000,
            len(audio) / 24000
        )

    start = active[0]
    end = active[-1]

    leading = start / 24000
    trailing = (
        len(audio) - end - 1
    ) / 24000

    return leading, trailing


def trim_silence(
    audio,
    threshold=0.005,
    keep_trailing=0.38
):

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    if audio.ndim > 1:
        audio = audio[:, 0]

    active = np.where(
        np.abs(audio) > threshold
    )[0]

    if len(active) == 0:
        return None

    start = active[0]
    end = active[-1] + 1

    # Keep a small amount of trailing silence
    keep_samples = int(
        keep_trailing * 24000
    )

    end = min(
        end + keep_samples,
        len(audio)
    )

    return audio[start:end]