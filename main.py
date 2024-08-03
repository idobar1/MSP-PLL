from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt 
from math_utils import *
import sounddevice as sd
from config import FiltType, Config
import math_utils
import open_audio
from scipy.io.wavfile import write as write_wav
from pydub.playback import play

class MathConfig:  # worked well
    loop_gain: float
    loop_filt_type: FiltType
    loop_filter_mem: int
    VCO_gain: float
    f0: float

    def __init__(self, loop_gain, loop_filt_type, loop_filter_mem, VCO_gain, f0):
        self.loop_gain = loop_gain
        self.loop_filt_type = loop_filt_type
        self.loop_filter_mem = loop_filter_mem
        self.VCO_gain = VCO_gain
        self.f0 = f0

    def config_to_str(self):
        return f"loop_gain * VCO_gain= {self.loop_gain*self.VCO_gain}, loop_filt_type = {self.loop_filt_type}, loop_filter_mem  = {self.loop_filter_mem}, f0 = {self.f0}"


def main():
    FileConfig = Config.FileConfig
    DebugConfig = Config.DebugConfig

    ### Input 1: Music signal ###
    fname = FileConfig.fname
    format = FileConfig.format
    
    audio, samples, num_of_channels, fs = open_audio.open_audio_file(fname, format)
    if num_of_channels == 2:
        left_ch, right_ch = open_audio.separate_channels(samples)
        processing_samples = left_ch
    else:
        processing_samples = samples

    cut_len_sec = len(processing_samples)
    # Cut audio (for debug) #
    if DebugConfig.cut_len_sec != 0:
        cut_start_sec = DebugConfig.cut_start_sec
        cut_len_sec = DebugConfig.cut_len_sec
        cut_stop_sec = cut_start_sec+cut_len_sec
        processing_samples = processing_samples[cut_start_sec * fs : fs * cut_stop_sec]
    #

    t = np.arange(0,cut_len_sec,1/fs)
    s_t = normalize_signal(processing_samples)
    O_t = math_utils.onset_func(s_t)
    e_t, x_t = math_utils.PLL(O_t, fs, math_config)
    metronome = metronome_thresholding(x_t)
    synched_metronome = math_utils.sync_metronome(metronome, O_t, fs, 1)

    sound_to_save, sound_metronome = add_metronome_with_sound(synched_metronome, s_t, t, fs)
    
    x_t_audio = AudioSegment(
    ((sound_to_save*32767/max(abs(sound_to_save))).astype('int16')).tobytes(),
    frame_rate=fs,
    sample_width=samples.dtype.itemsize, 
    channels=1) 
    

    x_t_audio.export("Music and Metronome.wav", format="wav")
    plt.figure()
    plt.plot(t, O_t)
    plt.plot(t, synched_metronome)
    plt.title('Onset Function & synched metronome')
    plt.show()

    ### Input 2: Sine Wave ###
    # sin_len = 5  # sec
    # f_sin = 2
    # fs = 44100
    # t = np.arange(0, sin_len, 1/fs)
    # sin_input = np.sin(2 * np.pi * f_sin * t)
    # sin_input_normalized = normalize_signal(sin_input)
    # e_t, x_t = math_utils.PLL(sin_input_normalized, fs)
    # metronome = metronome_thresholding(x_t)

    # plt.figure()
    # plt.plot(t, sin_input, label="sine input")
    # plt.plot(t, x_t, label="output")
    # plt.legend()
    # plt.title('Sine input & Output Signals')
    # plt.show()    
    # plt.figure()
    # plt.plot(t, e_t)
    # plt.title('e_t for sine input')
    # plt.show()

    ### Input 3: Square Wave, dc = 0.5 ###
    # sq_len = 10  # sec
    # f = 2
    # fs = 44100
    # t = np.arange(0, sq_len, 1/fs)
    # sq_01 = math_utils.synth_square(t, f, 0.5)
    # sin_input_normalized = normalize_signal(sq_01)
    # e_t, x_t = math_utils.PLL(sin_input_normalized, fs)

    # plt.figure()
    # plt.plot(t, sq_01, label="square wave input")
    # plt.plot(t, x_t, label="output")
    # plt.legend()
    # plt.title('Square Wave Input & Output Signals')
    # plt.show()    
    # plt.figure()
    # plt.plot(t, e_t)
    # plt.title('e_t for square wave (50%) input')
    # plt.show()

    ### Input 4: Square Wave, dc = 0.1 ###
    math_config_list = [  # TODO for each input we'll have this list of these configs. In the end we'll have 2 plots for each config
    MathConfig(loop_gain = 0.01, 
        loop_filt_type = FiltType.MA, 
        loop_filter_mem = 22050, 
        VCO_gain = 2*np.pi/100, 
        f0=1.8),
    MathConfig(loop_gain = 0., 
        loop_filt_type = FiltType.MA, 
        loop_filter_mem = 1000, 
        VCO_gain = 2*np.pi/100, 
        f0=1.8),
                            ]
    for math_config in math_config_list:
        sq_len = 10  # sec
        f = 2
        fs = 44100
        t = np.arange(0, sq_len, 1/fs)
        sq_01 = math_utils.synth_square(t, f, 0.1)
        sin_input_normalized = normalize_signal(sq_01)
        e_t, x_t = math_utils.PLL(sin_input_normalized, fs, math_config)

        plt.figure()
        plt.plot(t, sq_01, label="square wave input")
        plt.plot(t, x_t, label="output")
        plt.legend()
        plt.title('Square Wave Input & Output Signals')
        plt.show()    
        plt.figure()
        plt.plot(t, e_t)
        plt.title('e_t for square wave (10%) input')
        plt.show()
if __name__ == "__main__":
    main()