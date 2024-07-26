from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt 
from math_utils import *
import sounddevice as sd

from scipy.io.wavfile import write as write_wav
from pydub.playback import play

fname = "Heart Of Glass.wav"
format = 'wav'
fname = "Daft Punk - Around the World.mp3" 
format = 'mp3' 

audio = AudioSegment.from_file(fname,format=format)
samples = np.array(audio.get_array_of_samples())
channels = audio.channels
fs = audio.frame_rate

# seperate 2 channels
left_ch = samples[::2]
right_ch = samples[1::2]

# taking few secs only
# Could be done using pydub library also
#TODO: move to parameters dedicated file
cut_len_sec = 20
cut_start_sec = 0 # 60
cut_stop_sec = cut_start_sec+cut_len_sec
left_ch = left_ch[(cut_start_sec*fs):(fs*cut_stop_sec)] 
t = np.arange(0,cut_len_sec,1/fs)

s_t = left_ch/np.max(left_ch) #normalize to avoid overflow in processing
O_t = onset_func(s_t)
x_t = PLL(O_t,fs)

x_t_mod = np.multiply(x_t, np.cos(2*np.pi*440*t) + 0.25*np.cos(2*np.pi*440*3/2*t) + 0.2*np.cos(2*np.pi*440*5/4*t))
x_t_save = x_t_mod + s_t
x_t_audio = AudioSegment(
    ((x_t_save*32767/2).astype('int16')).tobytes(), 
    frame_rate=fs,
    sample_width=samples.dtype.itemsize, 
    channels=1)
x_t_audio.export("x_t_mod.wav", format="wav")

x_t_save_2 = x_t + s_t
# TODO: high pass filter for metronome sound,
#       fix code (order)
#       make synthetic inputs as required
#       
x_t_audio = AudioSegment(
    ((x_t*32767).astype('int16')).tobytes(), 
    frame_rate=fs,
    sample_width=samples.dtype.itemsize, 
    channels=1)
x_t_audio.export("x_t.wav", format="wav")

plt.figure(1)
plt.title("Welch PSD Estimation")
plot_est_spectrum(s_t,fs,4096)
plt.figure(2)
plt.plot(t,O_t)
plt.title('Onset Function')
plt.show()

plt.figure(3)
# plt.plot(t, x_t_mod)
plt.plot(t, x_t)
plt.title("x_t - metronome")
plt.show()
a=5