import numpy as np

def compute_volume(frame):
    return np.sqrt(np.mean(frame**2)) if len(frame) > 0 else 0.0

def compute_ste(frame):
    return np.sum(frame**2) / len(frame) if len(frame) > 0 else 0.0

def compute_zcr(frame):
    if len(frame) == 0:
        return 0.0
    zero_crossings = np.count_nonzero(np.diff(np.sign(frame)))
    return zero_crossings / len(frame)

def compute_sr(frame, vol_threshold=0.01, zcr_threshold=0.1):
    vol = compute_volume(frame)
    zcr = compute_zcr(frame)
    return 1 if (vol < vol_threshold and zcr < zcr_threshold) else 0

def compute_autocorr_f0(frame, fs, fmin=50, fmax=500):
    if len(frame) == 0:
        return 0
    frame = frame - np.mean(frame)
    corr = np.correlate(frame, frame, mode='full')
    corr = corr[len(corr)//2:]
    d = np.diff(corr)
    start = np.nonzero(d > 0)[0]
    if len(start) == 0:
        return 0
    lag = start[0]
    if lag == 0:
        return 0
    f0 = fs / lag
    if f0 < fmin or f0 > fmax:
        return 0
    return f0

def compute_amdf(frame):
    length = len(frame)
    if length == 0:
        return np.array([0])
    amdf = []
    for tau in range(length):
        diff_sum = 0.0
        for n in range(length - tau):
            diff_sum += abs(frame[n] - frame[n + tau])
        amdf.append(diff_sum / (length - tau))
    return np.array(amdf)

def compute_amdf_f0(frame, fs, fmin=50, fmax=500):
    length = len(frame)
    if length == 0:
        return 0
    frame = frame - np.mean(frame)
    amdf_values = compute_amdf(frame)

    min_lag = int(fs // fmax)
    max_lag = int(fs // fmin) if fmin != 0 else length // 2
    if max_lag > len(amdf_values):
        max_lag = len(amdf_values) - 1
    if min_lag < 1 or min_lag >= max_lag:
        return 0

    lag_range = np.arange(min_lag, max_lag)
    best_lag = lag_range[np.argmin(amdf_values[min_lag:max_lag])]
    if best_lag == 0:
        return 0
    f0 = fs / best_lag
    if f0 < fmin or f0 > fmax:
        return 0
    return f0
