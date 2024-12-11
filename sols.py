from filters import *
from utils import find_slope


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
