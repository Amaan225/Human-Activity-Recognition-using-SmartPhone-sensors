import numpy as np
from scipy.signal import butter, filtfilt

def highpass_filter(signal, fs=100, cutoff=0.3, order=3):
    
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, signal)
