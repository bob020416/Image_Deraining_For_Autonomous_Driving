from jax import numpy as np
from jax import jit


@jit
def PSNR(grount_truth, output):
    mse = ((grount_truth - output) ** 2).mean()
    return 10 * np.log10(255**2 / mse)


@jit
def SSIM(grount_truth, output):
    mux = np.mean(grount_truth)
    muy = np.mean(output)
    sigmax = np.std(grount_truth)
    sigmay = np.std(output)
    sigmaxy = np.cov(grount_truth.flatten(), output.flatten())[0, 1]
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    return (
        (2 * mux * muy + c1)
        * (2 * sigmaxy + c2)
        / ((mux**2 + muy**2 + c1) * (sigmax**2 + sigmay**2 + c2))
    )
