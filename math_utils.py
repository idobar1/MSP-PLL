import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt 
from config import Config
## TODO: Edit parameters to reasonable value when known
MathConfig = Config.MathConfig

def normalize_signal(samples):
    return samples/np.max(samples)

def synth_square(t_vec,f,d_c=0.5):
    return (ss.square(2 * np.pi * f * t_vec, d_c)*2-1)

def plot_est_spectrum(x,fs,nperseg=65536):
    f, Pxx_left = ss.welch(x,fs,nperseg=nperseg)
    plt.semilogy(f,Pxx_left)
    plt.grid()
    # plt.plot(f, Pxx_left)
    plt.show()
    
def onset_func(s_t):
    E = np.power(s_t, 2)
    dE_pos = np.diff(E, append=0)
    dE_pos[dE_pos < 0] = 0
    O_t = dE_pos - np.mean(dE_pos) #remove "DC"
    return O_t


 ##TODO: Debug
 ## Documnet the relation of T and the cutoff Frequency
 # Maybe add more complex "MA" Filter (with some decay)
def loop_filter(theta, t, filt_type, T=10):
    if(filt_type == "MA"):
        if(t < T):
            e_t = (1/(t + 1))*np.sum(theta[0:t+1]) * MathConfig.loop_gain
        else:
            e_t = (1/T)*np.sum(theta[t-T:t]) * MathConfig.loop_gain
    elif(filt_type == "gain"):
        e_t = theta[t] * MathConfig.loop_gain
    else:
        print("Invalid filt_type")
        e_t = 0
    return e_t
    
def phase_comp(O_t,x_t):
    theta_t = x_t*O_t
    return theta_t

def VCO(e_t, x_phase_t, VCO_gain):
    dphase = e_t * VCO_gain
    next_phase = x_phase_t + dphase
    return next_phase

def PLL(Onsets, fs):
    x = np.zeros(Onsets.shape) 
    x_phase = np.zeros(Onsets.shape)
    inst_f = np.zeros(Onsets.shape)
    theta = np.zeros(Onsets.shape) 
    e = np.zeros(Onsets.shape)
    #Random phase init
    x_phase[0] = 2*np.pi*np.random.rand()
    x[0] = np.sin(x_phase[0]) 
    
    #PLL Loop
    for n in range(len(Onsets)-1):
        theta[n] = phase_comp(Onsets[n],x[n])
        e[n] = loop_filter(theta, n, MathConfig.loop_filt_type, T=MathConfig.loop_filter_mem)      
        x_phase[n+1] = VCO(e[n],x_phase[n], MathConfig.VCO_gain) 
        inst_f[n+1]  = MathConfig.f0 + (x_phase[n+1] - x_phase[n])*fs/(2 * np.pi)     
        x[n+1] = np.sin(np.unwrap([2 * np.pi * MathConfig.f0 * n/fs + x_phase[n+1]])) 

        if(n % 10000 == 0): ## For debug
            print(f"{int(100*n/len(Onsets))}%")
    print("100%")

    return e, x

def zero_cross_detect(x):
    zc_vec = np.zeros(x.shape)
    zc_vec[x > 0] = 1
    zc_vec=np.diff(zc_vec)
    zc_vec[zc_vec < 0]=0
    return zc_vec

def metronome_thresholding(x,th=0.999):
    metronome = np.zeros(x.shape)
    metronome[x>th]=1
    return metronome
    
def find_metronome_delay(onsets,metronome):
    max_ix_onsets = np.argmax(onsets)
    possible_delays = (np.where(metronome>0)[0])-max_ix_onsets
    delay_ix = np.argmin(np.abs(possible_delays))
    delay = possible_delays[delay_ix]

    return delay
## TODO: use np.roll(metronome,delay)
