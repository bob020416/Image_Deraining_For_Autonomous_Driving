import numpy as np
from tqdm import tqdm
from filters import sector_region


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
    for op in tqdm(ops):
        I = op(I).astype(np.uint8)
        steps.append(I)
    return steps

def find_slope(I):
    I_freq = freq_domain(I) / 255

    def score(a):
        # print(a)
        score = 0
        region = sector_region(*I.shape, a, 2)
        score = np.sum(I_freq[region == 1])
        return score

    max_a, max_score = -1, -1
    for a in [*np.linspace(-10, -2, 8), *np.linspace(2, 10, 8)]:
        s = score(a)
        print(a, s)
        if s > max_score:
            max_a, max_score = a, s
    print(max_a, max_score)
    return max_a, max_score
