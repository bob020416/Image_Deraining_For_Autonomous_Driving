import numpy as np


def freq_domain(I):
    I_fft = np.fft.fft2(I)
    I_fft = np.fft.fftshift(I_fft)
    I_fft = np.log(1 + np.abs(I_fft))
    I_fft = np.uint8(I_fft * 255 / np.max(I_fft))
    return I_fft


def analyze(steps):
    ret = []
    for I in steps:
        ret.append(freq_domain(I))
    return ret


def chain(I, ops):
    steps = [I]
    for op in ops:
        I = op(I).astype(np.uint8)
        steps.append(I)
    return steps
