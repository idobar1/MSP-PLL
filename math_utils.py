import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt 

loop_gain = 1

def plot_est_spectrum(x,fs,nperseg=65536):
    f, Pxx_left = ss.welch(x,fs,nperseg=nperseg)
    plt.semilogy(f,Pxx_left)
    plt.grid()
    # plt.plot(f, Pxx_left)
    plt.show()
    
def onset_func(x):
    E = np.power(x,2)
    dE_pos = np.diff(E,append=0)
    dE_pos[dE_pos<0] = 0
    O_t = dE_pos-np.mean(dE_pos) #remove "DC"
    return O_t

def loop_filter(theta_t,t,type,T=10): ##TODO: Debug
    if(type == "MA"):
        e_t[t] = (1/T)*np.sum(theta_t[t-T:t]) * loop_gain
    elif(type == "gain"):
        e_t[t] = theta_t[-1] * loop_gain
    else:
        print("Invalid filt_type")
        e_t = 0
    return e_t
    
        
        
    