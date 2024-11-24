from filters import *
from utils import freq_domain


def sol1():
    return [
        lambda I: bilateral_filter(I, d=9, sigmaColor=75, sigmaSpace=75),
        lambda I: gaussian_sharpen(I, ksize=(5, 5), sigma=20, alpha=2, beta=-1),
    ]


def sol2():
    return [
        lambda I: line_filter(I, a=0.01, threshold=10, mul=0.5),
        lambda I: median_filter(I, ksize=3),
        lambda I: gamma_correction(I, gamma=0.5),
    ]


def sol3(I):
    # a, _ = find_slope(I)
    return [
        lambda I: sector_filter(I, a=5, threshold=3, mul=0, lratio=0, uratio=0.5),
        lambda I: median_filter(I, ksize=3),
    ]


def sol4():
    return [
        lambda I: median_filter(I, ksize=5),
        lambda I: gaussian_sharpen(I, ksize=(3, 3), sigma=20, alpha=2, beta=-1),
        lambda I: median_filter(I, ksize=3),
        lambda I: laplacian_filter(I) + I,
        lambda I: median_filter(I, ksize=3),
        lambda I: gaussian_sharpen(I, ksize=(3, 3), sigma=20, alpha=2, beta=-1),
        lambda I: gaussian_sharpen(I, ksize=(3, 3), sigma=20, alpha=2, beta=-1),
    ]


def find_slope(I):
    I_freq = freq_domain(I) / 255

    def score(a):
        # print(a)
        score = 0
        region = sector_region(*I.shape, a, 2)
        score = np.sum(I_freq[region == 1])
        return score

    (
        max_a,
        max_score,
    ) = (
        -1,
        -1,
    )
    for a in [*np.linspace(-15, -2, 10), *np.linspace(2, 15, 10)]:
        s = score(a)
        print(a, s)
        if s > max_score:
            max_a, max_score = a, s
    print(max_a, max_score)
    return max_a, max_score
