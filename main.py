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

    ## Input 1: Music signal ###
    
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

    plt.figure()
    plt.plot(t, s_t)
    plt.title('s(t) - the raw samples of the left channel')
    plt.show()
    
    math_config_list = [
    MathConfig(loop_gain = 1, 
        loop_filt_type = FiltType.MA, 
        loop_filter_mem = 20000, 
        VCO_gain = 2*np.pi/100, 
        f0=2.1)
                            ]
    for math_config in math_config_list:
        e_t, x_t = math_utils.PLL(O_t, fs, math_config)
        metronome = metronome_thresholding(x_t)
        synched_metronome = math_utils.sync_metronome(metronome, O_t, fs, 1)

        metronome_sound, song_w_metro = add_metronome_with_sound(synched_metronome, s_t, t, fs)
        song_w_metro_audio = AudioSegment(
        ((song_w_metro*32767/max(abs(song_w_metro))).astype('int16')).tobytes(),
        frame_rate=fs,
        sample_width=samples.dtype.itemsize, 
        channels=1) 
        

        song_w_metro_audio.export("Music and Metronome.wav", format="wav")
        plt.figure()
        plt.plot(t, O_t)
        plt.plot(t, synched_metronome)
        plt.title('Onset Function & synched metronome')
        plt.show()
        
        plt.figure()
        plt.plot(t, metronome_sound)
        plt.title('metronome sound')
        plt.show()
        
        plt.figure()
        plt.plot(t, e_t)
        plt.title('e_t for Another One bites The Dust input')
        plt.show()

    ### Input 2: Sine Wave ###
    
    # math_config_list = [
    # MathConfig(loop_gain = 0.0025, 
    #     loop_filt_type = FiltType.GAIN, 
    #     loop_filter_mem = 20000, 
    #     VCO_gain = 2*np.pi/100, 
    #     f0=2)
    #                         ]
    # sin_len = 10  # sec
    # f_input = 1.4
    # fs = 44100
    # t = np.arange(0, sin_len, 1/fs)
    # sin_input = np.sin(2 * np.pi * f_input * t)
    # plt.figure()
    # plt.plot(t, sin_input)
    # plt.title('s(t) - sine wave')
    # plt.show()
    # O_t = math_utils.onset_func(sin_input) 
    # plt.figure()
    # plt.plot(t, O_t)
    # plt.title('O(t) - Onsets of sine wave')
    # plt.show()
    # for math_config in math_config_list:
    #     e_t, x_t = math_utils.PLL(sin_input, fs, math_config)   
    #     plt.figure()
    #     plt.plot(t, sin_input, label="sine input")
    #     plt.plot(t, x_t, label="output")
    #     plt.legend()
    #     plt.title('Sine input & Output Signals\n' + math_config.config_to_str() + '\nf_input = ' + str(f_input))
    #     plt.show()    
    #     plt.figure()
    #     plt.plot(t, e_t)
    #     plt.title('e_t for sine input\n' + math_config.config_to_str() + '\nf_input = ' + str(f_input))
    #     plt.show()

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
    # math_config_list = [
    # MathConfig(loop_gain = 10, 
    #     loop_filt_type = FiltType.GAIN, 
    #     loop_filter_mem = 44100, 
    #     VCO_gain = 2*np.pi/100, 
    #     f0=1.9),
    # MathConfig(loop_gain = 0.0001, 
    #     loop_filt_type = FiltType.MA, 
    #     loop_filter_mem = 1000, 
    #     VCO_gain = 2*np.pi/100, 
    #     f0=1.8),
                            # ]
    # for math_config in math_config_list:
    #     sq_len = 10  # sec
    #     f = 2
    #     fs = 44100
    #     t = np.arange(0, sq_len, 1/fs)
    #     sq_01 = math_utils.synth_square(t, f, 0.1)
    #     s_t =  normalize_signal(sq_01)
    #     plt.figure()
    #     plt.plot(t, s_t)
    #     plt.title('s(t) - square wave 0.1')
    #     plt.show()
    #     O_t = math_utils.onset_func(s_t)
    #     plt.figure()
    #     plt.plot(t, O_t)
    #     plt.title('O(t) - Onsets of square wave 0.1')
    #     plt.show()
    #     e_t, x_t = math_utils.PLL(O_t, fs, math_config)

    #     plt.figure()
    #     plt.plot(t, sq_01, label="square wave input")
    #     plt.plot(t, x_t, label="output")
    #     plt.legend()
    #     plt.title('Square Wave Input & Output Signals\n' + math_config.config_to_str())
    #     plt.show()    
    #     plt.figure()
    #     plt.plot(t, e_t)
    #     plt.title('e_t for square wave (10%) input\n' + math_config.config_to_str())
    #     plt.show()

    ### Input 5:  2 Freqs Sine Wave ###
    
    # math_config_list = [
    #     MathConfig(
    #         loop_gain = 0.002,
    #         loop_filt_type = FiltType.MA, 
    #         loop_filter_mem = 20000, 
    #         VCO_gain = 2*np.pi/100, 
    #         f0=2
    #     ),
    #     MathConfig(
    #         loop_gain = 0.002, 
    #         loop_filt_type = FiltType.GAIN, 
    #         loop_filter_mem = 20000, 
    #         VCO_gain = 2*np.pi/100, 
    #         f0=2
    #     )
    # ]
    # sin1_len = 10  # sec
    # sin2_len = 10  # sec
    # f1_input = 1.8
    # f2_input = 2.3
    # fs = 44100
    # two_freqs_sin_input = math_utils.two_freqs_sines(f1_input, sin1_len, f2_input, sin2_len, fs)
    # t = np.arange(0, sin1_len+sin2_len, 1/fs)
    # plt.figure()
    # plt.plot(t, two_freqs_sin_input)
    # plt.title(f's(t) - two freqs sine wave - {f1_input}[Hz] and {f2_input}[Hz]')
    # plt.show()
    # O_t = math_utils.onset_func(two_freqs_sin_input) 
    # plt.figure()
    # plt.plot(t, O_t)
    # plt.title('O(t) - Onsets of two freqs sine wave')
    # plt.show()
    # for math_config in math_config_list:
    #     e_t, x_t = math_utils.PLL(two_freqs_sin_input, fs, math_config)   
    #     plt.figure()
    #     plt.plot(t, two_freqs_sin_input, label="two freqs sine input")
    #     plt.plot(t, x_t, label="output")
    #     plt.legend()
    #     plt.title('Two Freqs Sine input & Output Signals\n' + math_config.config_to_str() + '\nf1_input = ' + str(f1_input)+ '\nf2_input = ' + str(f2_input))
    #     plt.show()    
    #     plt.figure()
    #     plt.plot(t, e_t)
    #     plt.title('e_t for two freqs sine input\n' + math_config.config_to_str() + '\nf1_input = ' + str(f1_input) + '\nf2_input = ' + str(f2_input))
    #     plt.show()

    ### Input 6:  2 Freqs Square Wave ###
    
    # math_config_list = [
    #     MathConfig(
    #         loop_gain = 0.004,
    #         loop_filt_type = FiltType.MA, 
    #         loop_filter_mem = 10000, 
    #         VCO_gain = 2*np.pi/100, 
    #         f0=2
    #     ),
    #     MathConfig(
    #         loop_gain = 0.004, 
    #         loop_filt_type = FiltType.GAIN, 
    #         loop_filter_mem = 10000, 
    #         VCO_gain = 2*np.pi/100, 
    #         f0=2
    #     )
    # ]
    # sin1_len = 10  # sec
    # sin2_len = 10  # sec
    # f1_input = 1.8
    # f2_input = 2.2
    # fs = 44100
    # two_freqs_sin_input = math_utils.two_freqs_squares(f1_input, sin1_len, f2_input, sin2_len, fs)
    # t = np.arange(0, sin1_len+sin2_len, 1/fs)
    # plt.figure()
    # plt.plot(t, two_freqs_sin_input)
    # plt.title(f's(t) - two freqs square wave - {f1_input}[Hz] and {f2_input}[Hz]')
    # plt.show()
    # O_t = math_utils.onset_func(two_freqs_sin_input) 
    # plt.figure()
    # plt.plot(t, O_t)
    # plt.title('O(t) - Onsets of two freqs square wave ')
    # plt.show()
    # for math_config in math_config_list:
    #     e_t, x_t = math_utils.PLL(two_freqs_sin_input, fs, math_config)   
    #     plt.figure()
    #     plt.plot(t, two_freqs_sin_input, label="two freqs square wave input")
    #     plt.plot(t, x_t, label="output")
    #     plt.legend()
    #     plt.title('Two Freqs square wave input & Output Signals\n' + math_config.config_to_str() + '\nf1_input = ' + str(f1_input)+ '\nf2_input = ' + str(f2_input))
    #     plt.show()    
    #     plt.figure()
    #     plt.plot(t, e_t)
    #     plt.title('e_t for two freqs square wave input\n' + math_config.config_to_str() + '\nf1_input = ' + str(f1_input) + '\nf2_input = ' + str(f2_input))
    #     plt.show()
        


if __name__ == "__main__":
    main()