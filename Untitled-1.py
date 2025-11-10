# %%
import numpy as np

import soundfile as sf

import mbrola

# Create an MBROLA object (replace 'path/to/mbrola/voice' with your actual voice file)
synth = mbrola.('C:/Users/odhayaa/Downloads/us2')

# Synthesize a word or phrase with phonemes and pitch/duration information
# (This example is simplified; actual usage requires precise phoneme and prosodic data)
synth.synthesize([('h', 100, 100), ('a', 200, 120), ('u', 150, 110), ('s', 200, 100)])

# Save the synthesized speech to a WAV file
synth.save('output.wav')

# Phone frequencies (simplified formant frequencies for demonstration)
PHONE_FORMANTS = {
    "AA0": [700, 1100], "AE0": [660, 1700], "AH0": [600, 1040], "AO0": [400, 750],
    "IY0": [300, 2200], "UH0": [440, 1020], "EH0": [530, 1850], "OW0": [400, 1000],
    "ER0": [500, 1500], "AY0": [700, 1800], "AW0": [700, 1200],
    # add more phones here...
}

SAMPLE_RATE = 16000
DURATION = 0.15  # 150ms per phone

def synthesize_phone(phone):
    """
    Generate a simple vowel-like sound using two sine waves for formants.
    Consonants are generated as noise bursts or silence.
    """
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    if phone in PHONE_FORMANTS:
        f1, f2 = PHONE_FORMANTS[phone]
        wave = 0.5 * np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)
    elif phone in {"B","D","G","K","P","T"}:  # plosives
        wave = np.random.randn(len(t)) * 0.2
    elif phone in {"S","SH","F","V","Z","ZH"}:  # fricatives
        wave = np.random.randn(len(t)) * 0.1
    else:
        wave = np.zeros_like(t)
    # Normalize
    wave /= np.max(np.abs(wave)) + 1e-9
    return wave

def synthesize_sequence(phones):
    """
    Concatenate all phone waveforms into a single waveform.
    """
    audio = np.concatenate([synthesize_phone(p) for p in phones])
    return audio

if __name__ == "__main__":
    # Example phone sequence from your model
    # phone_seq = ["HH", "EH0", "L", "OW0"]  # hello
    phone_seq = ["DH", "AH0", "K", "AE0", "T"]  # the cat
    audio = synthesize_sequence(phone_seq)
    
    # Write to WAV
    sf.write("/home/odhayaa/02/MA416_modernNETtalk/synthesize.wav", audio, SAMPLE_RATE)



