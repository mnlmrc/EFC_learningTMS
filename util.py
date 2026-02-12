import numpy as np
import PcmPy as pcm

def prew_res_from_part_mean(data, part_vec):
    T, C = data.shape
    X = pcm.indicator(part_vec)
    beta, *_ = np.linalg.lstsq(X, data)
    err = data - X @ beta
    cov = (err.T @ err) / err.shape[0]
    return data / np.sqrt(np.diag(cov))