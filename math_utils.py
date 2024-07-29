from strenum import StrEnum
import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt

## TODO: Move parameters to a file with all of the program parameters (also related w/ filename etc..)
## TODO: Edit parameters to reasonable value when known


class FiltType(StrEnum):
    MA = "MA"
    GAIN = "GAIN"


loop_gain       = 0.5
loop_filt_type  = FiltType.MA
loop_filter_mem = 20500
VCO_gain        = 2*np.pi/100 
f0              = 2.5 #[Hz]

def synth_square(t_vec,f,d_c=0.5):
    return ss.square(2 * np.pi * f * t_vec, d_c)

def plot_est_spectrum(x,fs,nperseg=65536):
    f, Pxx_left = ss.welch(x,fs,nperseg=nperseg)
    plt.semilogy(f,Pxx_left)
    plt.grid()
    # plt.plot(f, Pxx_left)
    plt.show()
    
def onset_func(s_t):
    """
    TODO: add documentation to all functions"""
    E = np.power(s_t,2)
    dE_pos = np.diff(E,append=0)
    dE_pos[dE_pos<0] = 0
    O_t = dE_pos-np.mean(dE_pos) #remove "DC"
    # O_t = dE_pos # Without removing the DC
    return O_t


 ##TODO: Debug
 ## Documnet the relation of T and the cutoff Frequency
 # Maybe add more complex "MA" Filter (with some decay)
def loop_filter(theta,t,filt_type,T=10):
    if(filt_type == FiltType.MA):
        if(t<T):
            e_t = (1/(t+1))*np.sum(theta[0:t+1]) * loop_gain
        else:
            e_t = (1/T)*np.sum(theta[t-T:t]) * loop_gain
    elif(filt_type == FiltType.GAIN):
        e_t = theta[t] * loop_gain
    else:
        print("Invalid filt_type")
        e_t = 0
    return e_t
    
def phase_comp(O_t,x_t):
    theta_t = x_t*O_t
    return theta_t

def VCO(e_t,x_phase_t,VCO_gain):
    dphase = e_t*VCO_gain
    next_phase = x_phase_t + dphase
    return next_phase

def PLL(Onsets,fs): ## TODO: Implement
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
        e[n] = loop_filter(theta,n,loop_filt_type,T=loop_filter_mem)     
        x_phase[n+1] = VCO(e[n],x_phase[n], VCO_gain) 
        inst_f[n+1]  = f0 + + (x_phase[n+1] - x_phase[n])*fs/(2*np.pi)    
        x[n+1] = np.sin(np.unwrap([2*np.pi*f0*n/fs+x_phase[n+1]]))       
        if(n%10000==0): ## For debug
            print(n/10000)
    return e, x

def zero_cross_detect(x):
    zc_vec = np.zeros(x.shape)
    zc_vec[x>0] = 1
    zc_vec=np.diff(zc_vec)
    zc_vec[zc_vec<0]=0
    return zc_vec

def metronome_thresholding(x,th=0.999):
    metronome = np.zeros(x.shape)
    metronome[x>th]=1
    return metronome
    
    

    