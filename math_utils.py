import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt 
from config import Config

def normalize_signal(samples):
    return samples/np.max(samples)

def synth_square(t_vec, f, d_c=0.5):
    return ((ss.square(2 * np.pi * f * t_vec, d_c) + 1)/2)

def jumping_tempo_square():
    pass

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
def loop_filter(theta, t, filt_type, math_config, T=10):
    if(filt_type == "MA"):
        if(t < T):
            e_t = (1/(t + 1))*np.sum(theta[0:t+1]) * math_config.loop_gain
        else:
            e_t = (1/T)*np.sum(theta[t-T:t]) * math_config.loop_gain
    elif(filt_type == "gain"):
        e_t = theta[t] * math_config.loop_gain
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

def PLL(pll_input, fs, math_config):
    x = np.zeros(pll_input.shape) 
    x_phase = np.zeros(pll_input.shape)
    inst_f = np.zeros(pll_input.shape)
    theta = np.zeros(pll_input.shape) 
    e = np.zeros(pll_input.shape)
    #Random phase init
    x_phase[0] = 2*np.pi*np.random.rand()
    x[0] = np.sin(x_phase[0]) 
    
    #PLL Loop
    for n in range(len(pll_input)-1):
        theta[n] = phase_comp(pll_input[n],x[n])
        e[n] = loop_filter(theta, n, math_config.loop_filt_type, math_config, T=math_config.loop_filter_mem)      
        x_phase[n+1] = VCO(e[n],x_phase[n], math_config.VCO_gain) 
        inst_f[n+1]  = math_config.f0 + (x_phase[n+1] - x_phase[n])*fs/(2 * np.pi)     
        x[n+1] = np.sin(np.unwrap([2 * np.pi * math_config.f0 * n/fs + x_phase[n+1]])) 

        if(n % 10000 == 0): ## For debug
            print(f"{int(100*n/len(pll_input))}%")
    print("100%")

    return e, x

def zero_cross_detect(x):
    zc_vec = np.zeros(x.shape)
    zc_vec[x > 0] = 1
    zc_vec=np.diff(zc_vec)
    zc_vec[zc_vec < 0]=0
    return zc_vec

def metronome_thresholding(x, th=0.999):
    metronome = np.zeros(x.shape)
    metronome[x>th]=1
    return metronome
    
def find_metronome_delay(metronome, onsets):
    max_ix_onsets = np.argmax(onsets)
    possible_delays = (np.where(metronome>0)[0])-max_ix_onsets
    delay_ix = np.argmin(np.abs(possible_delays))
    delay = possible_delays[delay_ix]

    return delay

def sync_metronome(metronome, onset, fs, num_of_sec_trim):
    start_val = int(fs*num_of_sec_trim)
    metronome_delay = find_metronome_delay(metronome[-start_val:], onset[-start_val:])
    metronome_synched = np.roll(metronome, -metronome_delay)
    return metronome_synched

def add_metronome_with_sound(metronome, signal, t, fs):
    m_t = np.convolve(metronome, np.ones((int(fs*0.1))), mode="same" )
    m_t_mod = np.multiply(m_t, np.cos(2*np.pi*440*t) + 0.25*np.cos(2*np.pi*440*3/2*t) + 0.1*np.cos(2*np.pi*440*5/4*t))
    m_t_mod = 2*m_t_mod/np.max(np.abs(m_t_mod))
    x_t_save_2 = m_t_mod + signal
    return m_t_mod, x_t_save_2