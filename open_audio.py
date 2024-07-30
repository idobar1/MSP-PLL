from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt 
from math_utils import *
import sounddevice as sd

from scipy.io.wavfile import write as write_wav
from pydub.playback import play

def open_audio_file(fname, format):
    audio = AudioSegment.from_file(fname,format=format)
    samples = np.array(audio.get_array_of_samples())
    num_of_channels = audio.channels
    fs = audio.frame_rate

    return audio, samples, num_of_channels, fs

def separate_channels(audio_samples):
    return audio_samples[::2], audio_samples[1::2]