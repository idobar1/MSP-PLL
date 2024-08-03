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
    fname = FileConfig.fname
    format = FileConfig.format

    audio, samples, num_of_channels, fs = open_audio.open_audio_file(fname, format)

    left_ch, right_ch = open_audio.separate_channels(samples)
    
    processing_samples = left_ch
    cut_len_sec = len(processing_samples)
    ### Cut audio (for debug) ###
    if DebugConfig.cut_len_sec != 0:
        cut_start_sec = DebugConfig.cut_start_sec
        cut_len_sec = DebugConfig.cut_len_sec
        cut_stop_sec = cut_start_sec+cut_len_sec
        processing_samples = processing_samples[cut_start_sec * fs : fs * cut_stop_sec]
    ###

    t = np.arange(0,cut_len_sec,1/fs)
    s_t = normalize_signal(processing_samples)
    O_t = math_utils.onset_func(s_t)
    e_t, x_t = math_utils.PLL(O_t, fs)
    metronome = metronome_thresholding(x_t)
    synched_metronome = math_utils.sync_metronome(metronome, O_t, fs, 1)

    m_t = np.convolve(synched_metronome, np.ones((int(fs*0.1))), mode="same" )
    m_t_mod = np.multiply(m_t, np.cos(2*np.pi*440*t) + 0.25*np.cos(2*np.pi*440*3/2*t) + 0.1*np.cos(2*np.pi*440*5/4*t))
    
    m_t_mod = 2*m_t_mod/np.max(np.abs(m_t_mod))
    x_t_save_2 = m_t_mod + s_t
    
    x_t_audio = AudioSegment(
    ((x_t_save_2*32767/max(abs(x_t_save_2))).astype('int16')).tobytes(),
    frame_rate=fs,
    sample_width=samples.dtype.itemsize, 
    channels=1) 
    

    x_t_audio.export("song_w_metro.wav", format="wav")
    plt.figure(1)
    plt.title("Welch PSD Estimation")
    plot_est_spectrum(s_t,fs,4096)
    plt.figure(2)
    plt.plot(t,O_t)
    plt.title('Onset Function')
    plt.show()
    plt.figure(3)
    # plt.plot(t, x_t_mod)
    plt.plot(t, metronome)
    plt.title("metronome")
    plt.show()
    plt.plot(t, synched_metronome)
    plt.title("synched_metronome")
    plt.show()
    plt.plot(t, x_t)
    plt.title("x_t - metronome")
    plt.show()


if __name__ == "__main__":
    main()