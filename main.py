from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt 
from math_utils import *
import sounddevice as sd
from config import Config as config
import math_utils
import open_audio
from scipy.io.wavfile import write as write_wav
from pydub.playback import play

def main():
    FileConfig = config.FileConfig
    DebugConfig = config.DebugConfig

    ### Input 1: Music signal ###
    # fname = FileConfig.fname
    # format = FileConfig.format

    # audio, samples, num_of_channels, fs = open_audio.open_audio_file(fname, format)
    # if num_of_channels == 2:
    #     left_ch, right_ch = open_audio.separate_channels(samples)
    #     processing_samples = left_ch
    # else:
    #     processing_samples = samples

    # cut_len_sec = len(processing_samples)
    # # Cut audio (for debug) #
    # if DebugConfig.cut_len_sec != 0:
    #     cut_start_sec = DebugConfig.cut_start_sec
    #     cut_len_sec = DebugConfig.cut_len_sec
    #     cut_stop_sec = cut_start_sec+cut_len_sec
    #     processing_samples = processing_samples[cut_start_sec * fs : fs * cut_stop_sec]
    # #

    # t = np.arange(0,cut_len_sec,1/fs)
    # s_t = normalize_signal(processing_samples)
    # O_t = math_utils.onset_func(s_t)
    # e_t, x_t = math_utils.PLL(O_t, fs)
    # metronome = metronome_thresholding(x_t)
    # synched_metronome = math_utils.sync_metronome(metronome, O_t, fs, 1)

    # sound_to_save, sound_metronome = add_metronome_with_sound(synched_metronome, s_t, t, fs)
    
    # x_t_audio = AudioSegment(
    # ((sound_to_save*32767/max(abs(sound_to_save))).astype('int16')).tobytes(),
    # frame_rate=fs,
    # sample_width=samples.dtype.itemsize, 
    # channels=1) 
    

    # x_t_audio.export("Music and Metronome.wav", format="wav")
    # plt.figure()
    # plt.plot(t, O_t)
    # plt.plot(t, synched_metronome)
    # plt.title('Onset Function & synched metronome')
    # plt.show()

    ### Input 2: Sine Wave ###
    sin_len = 5  # sec
    f_sin = 2
    fs = 44100
    t = np.arange(0, sin_len, 1/fs)
    sin_input = np.sin(2 * np.pi * f_sin * t)
    sin_input_normalized = normalize_signal(sin_input)
    e_t, x_t = math_utils.PLL(sin_input_normalized, fs)
    metronome = metronome_thresholding(x_t)

    plt.figure()
    plt.plot(t, sin_input, label="sine input")
    plt.plot(t, x_t, label="output")
    plt.legend()
    plt.title('Sine input & Output Signals')
    plt.show()    
    plt.figure()
    plt.plot(t, e_t)
    plt.title('e_t for sine input')
    plt.show()

if __name__ == "__main__":
    main()