import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt 

loop_gain = 3 ## TODO: Edit to reasonable value when known

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


 ##TODO: Debug
 ## Documnet the relation of T and the cutoff Frequency
 # Maybe add more complex "MA" Filter (with some decay)
def loop_filter(theta_t,t,type,T=10):
    if(type == "MA"):
        if(t<T):
            e_t = (1/(t+1))*np.sum(theta_t[0:t]) * loop_gain
        else:
            e_t = (1/T)*np.sum(theta_t[t-T:t]) * loop_gain
    elif(type == "gain"):
        e_t = theta_t[t] * loop_gain
    else:
        print("Invalid filt_type")
        e_t = 0
    return e_t
    
        
        
    