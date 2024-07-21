from pydub import AudioSegment
import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt 
from utils import *

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

left_ch = left_ch/np.max(left_ch) #normalize to avoid overflow in processing

plt.figure(1)
plt.title("Welch PSD Estimation")
plot_est_spectrum(left_ch,fs,4096)


plt.figure(2)
O_t = onset_func(left_ch)
plt.plot(t,O_t)
plt.title('Onset Function')
plt.show()
a=5