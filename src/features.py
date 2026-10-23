import numpy as np
from python_speech_features import mfcc, delta

def extract_mfcc_39(signal, sample_rate, winlen, winstep, num_cep,n_fft, n_filt=26, preemph=0.97, ceplifter=22, normalize=False):
    base = mfcc(signal, samplerate = sample_rate, winlen = winlen, winstep = winstep,
                numcep = num_cep, nfilt = n_filt, nfft = n_fft, appendEnergy = True, 
                preemph = preemph, ceplifter=22)
    delta_1 = delta(base, 2)
    delta_2 = delta(delta_1, 2)
    
    features = np.concatenate([base, delta_1, delta_2], axis=1)

    # per-recording mean/variance normalization is OFF by default: for a single sustained
    # vowel the mean spectrum over frames IS the vowel identity, so subtracting it removes
    # the class information (closed-speaker accuracy 38-57% with it vs ~84% without)
    if normalize:
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True)
        std[std < 1e-8] = 1.0
        features = (features - mean) / std
    
    if features.ndim != 2 or features.shape[1] != 39:
        raise ValueError(f"Feature shape must be (n_frames, 39), got {features.shape}")
        
    return features