from pydub import AudioSegment
import numpy as np
from math_utils import *

def open_audio_file(fname, format):
    audio = AudioSegment.from_file(fname,format=format)
    samples = np.array(audio.get_array_of_samples())
    num_of_channels = audio.channels
    fs = audio.frame_rate

    return audio, samples, num_of_channels, fs

def separate_channels(audio_samples):
    return audio_samples[::2], audio_samples[1::2]