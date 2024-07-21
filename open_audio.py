from pydub import AudioSegment
import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt 

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

# taking the first minute only
# Could be done using pydub library also
left_ch = left_ch[:(fs*60)] 
t = np.arange(0,60,1/fs)

left_ch = left_ch/np.max(left_ch)
# f, Pxx_left = ss.welch(left_ch,fs,nperseg=65536)
# plt.plot(f, Pxx_left)
# plt.show()
def onset_func(x):
    E = np.power(x,2)
    dE_pos = np.diff(E,append=0)
    dE_pos[dE_pos<0] = 0
    O_t = dE_pos-np.mean(dE_pos)
    return O_t
# E_left = np.power(left_ch,2)
# dE_left_plus = np.diff(E_left,append=0)
# dE_left_plus[dE_left_plus<0] = 0
# O_t = dE_left_plus
O_t = onset_func(left_ch)
plt.plot(t,O_t)
plt.show()
a=5