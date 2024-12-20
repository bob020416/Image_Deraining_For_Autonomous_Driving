import numpy as np
import cv2


def notch_filter(I, threshold, mul=0.7):
    I_fft = np.fft.fft2(I)
    I_fft = np.fft.fftshift(I_fft)
    rows, cols = I.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    mask[crow - threshold : crow + threshold, ccol - threshold : ccol + threshold] = 0.7
    I_fft = I_fft * mask
    I_fft = np.fft.ifftshift(I_fft)
    I = np.fft.ifft2(I_fft)
    I = np.abs(I)
    return I


def gauss_filter(I, threshold, mul=0.7):
    I_fft = np.fft.fft2(I)
    I_fft = np.fft.fftshift(I_fft)
    rows, cols = I.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow - threshold : crow + threshold, ccol - threshold : ccol + threshold] = 0.7
    I_fft = I_fft * mask
    I_fft = np.fft.ifftshift(I_fft)
    I = np.fft.ifft2(I_fft)
    I = np.abs(I)
    return I


def high_pass_filter(I, threshold, mul=0.7):
    I_fft = np.fft.fft2(I)
    I_fft = np.fft.fftshift(I_fft)
    rows, cols = I.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow - threshold : crow + threshold, ccol - threshold : ccol + threshold] = mul
    I_fft = I_fft * mask
    I_fft = np.fft.ifftshift(I_fft)
    I = np.fft.ifft2(I_fft)
    I = np.abs(I)
    return I


def column_filter(I, threshold, mul=0.7):
    I_fft = np.fft.fft2(I)
    I_fft = np.fft.fftshift(I_fft)
    rows, cols = I.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.float32)
    mask[:, ccol - threshold : ccol + threshold] = mul
    I_fft = I_fft * mask
    I_fft = np.fft.ifftshift(I_fft)
    I = np.fft.ifft2(I_fft)
    I = np.abs(I)
    I = np.uint8(I)
    return I


def row_filter(I, threshold, mul=0.7):
    I_fft = np.fft.fft2(I)
    I_fft = np.fft.fftshift(I_fft)
    rows, cols = I.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.float32)
    mask[crow - threshold : crow + threshold, :] = mul
    I_fft = I_fft * mask
    I_fft = np.fft.ifftshift(I_fft)
    I = np.fft.ifft2(I_fft)
    I = np.abs(I)
    I = np.uint8(I)
    return I


def line_filter(I, a, threshold, mul=0.7):
    I_fft = np.fft.fft2(I)
    I_fft = np.fft.fftshift(I_fft)
    rows, cols = I.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.float32)
    for i in range(rows):
        for j in range(cols):
            x = j - ccol
            y = -i + crow
            if np.abs(a * x - y) / np.sqrt(a**2 + 1) < threshold:
                mask[i, j] = mul

    viz_mask = np.uint8(mask * 255)
    # cv2.imwrite("output/line_mask.jpg", viz_mask)

    I_fft = I_fft * mask
    I_fft = np.fft.ifftshift(I_fft)
    I = np.fft.ifft2(I_fft)
    I = np.abs(I)
    I = np.uint8(I)
    return I


def sector_region(rows, cols, a, threshold, lratio=0.2, uratio=0.8):
    crow, ccol = rows // 2, cols // 2
    # rub, cub = int(rows // 2 * uratio), int(cols // 2 * uratio)
    # rlb, clb = int(rows // 2 * lratio), int(cols // 2 * lratio)
    rub = np.array(rows // 2 * uratio, dtype=int)
    cub = np.array(cols // 2 * uratio, dtype=int)
    rlb = np.array(rows // 2 * lratio, dtype=int)
    clb = np.array(cols // 2 * lratio, dtype=int)
    mask = np.zeros((rows, cols), np.float32)
    for i in range(crow - rub, crow + rub):
        y = crow - i
        for j in range(ccol - cub, ccol + cub):
            x = j - ccol
            if x > -clb and x < clb:
                continue
            if np.rad2deg(abs(np.arctan2(y, x) - np.deg2rad(a))) < threshold:
                mask[i, j] = 1
                mask[rows - i, cols - j] = 1
    return mask


def sector_gauss(rows, cols, a, threshold):
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.float32)
    for i in range(rows):
        for j in range(ccol, cols):
            x = j - ccol
            y = -i + crow
            diff = 0.95 * np.rad2deg(abs(np.arctan2(y, x) - np.deg2rad(a)))
            dist = 0.0 * np.sqrt(x**2 + y**2)
            gauss = (
                np.exp(-((diff**2 + dist**2) / (2 * threshold**2)))
                / (2 * np.pi * threshold**2)
                * 50
            )
            mask[i, j] = gauss
            mask[-i, cols - j] = gauss

    # cv2.imwrite("output/sector_region.jpg", np.uint8(mask * 255))
    return mask


def sector_filter(I, a, threshold, mul=0.7, lratio=0.2, uratio=0.8):
    I_fft = np.fft.fft2(I)
    I_fft = np.fft.fftshift(I_fft)
    rows, cols = I.shape
    mask = 1 - sector_region(rows, cols, a, threshold, lratio, uratio) * (1 - mul)
    viz_mask = np.uint8(mask * 255)
    # cv2.imwrite("output/sector_mask.jpg", viz_mask)
    I_fft = I_fft * mask
    I_fft = np.fft.ifftshift(I_fft)
    I = np.fft.ifft2(I_fft)
    I = np.abs(I)
    I = np.uint8(I)
    return I


def sector_gauss_filter(I, a, threshold, mul=0.7, lratio=0.2, uratio=0.8):
    I_fft = np.fft.fft2(I)
    I_fft = np.fft.fftshift(I_fft)
    rows, cols = I.shape
    rows = np.array(rows)
    mask = 1 - sector_gauss(rows, cols, a, threshold)
    viz_mask = np.uint8(mask * 255)
    # cv2.imwrite("output/sector_mask.jpg", viz_mask)
    I_fft = I_fft * mask
    I_fft = np.fft.ifftshift(I_fft)
    I = np.fft.ifft2(I_fft)
    I = np.abs(I)
    I = np.uint8(I)
    return I


def slope_filter(I, a, kernel_size=5):
    rows, cols = I.shape
    crow, ccol = rows // 2, cols // 2
    kernel = np.zeros((kernel_size, kernel_size), np.float32)
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = j - kernel_size // 2
            y = -i + kernel_size // 2
            diff = np.abs(np.rad2deg(np.arctan2(y, x) - np.deg2rad(a)))
            kernel[i, j] = np.exp(-((diff**2) / (2 * 5**2))) / (2 * np.pi * 5**2)
    # cv2.imwrite("output/slope_kernel.jpg", np.uint8(kernel * 255))
    kernel = kernel / np.sum(kernel)
    kernel = np.array(kernel)
    return cv2.filter2D(I, -1, kernel)


def gamma_correction(I, gamma=1.0):
    I = I / 255.0
    I = cv2.pow(I, gamma)
    I = np.uint8(I * 255)
    return I


def histogram_equalization(I):
    I = cv2.equalizeHist(I)
    return I


def brightness_contrast(I, alpha=1.0, beta=0):
    I = cv2.convertScaleAbs(I, alpha=alpha, beta=beta)
    return I


def bilateral_filter(I, d=9, sigmaColor=75, sigmaSpace=75):
    I = cv2.bilateralFilter(I, d, sigmaColor, sigmaSpace)
    return I


def laplacian_filter(I):
    I = cv2.Laplacian(I, cv2.CV_64F)
    # I = cv2.convertScaleAbs(I)
    return I


def gaussian_filter(I, ksize=(5, 5), sigma=0):
    I = cv2.GaussianBlur(I, ksize, sigma)
    return I


def gaussian_sharpen(I, ksize=(5, 5), sigma=5, alpha=1.5, beta=-0.5):
    blurred = cv2.GaussianBlur(I, ksize, sigma)
    I = cv2.addWeighted(I, alpha, blurred, beta, 0)
    return I


def guided_filter(I, p, r=9, eps=1e-8):
    I = cv2.ximgproc.guidedFilter(p, I, r, eps)
    return I


def histogram_equalization_tar(I, target):
    I = cv2.equalizeHist(I, target)
    return I


def local_histogram_equalization(I, clipLimit=2.0, tileGridSize=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit, tileGridSize)
    I = clahe.apply(I)
    return I


def median_filter(I, ksize=5):
    I = cv2.medianBlur(I, ksize)
    return I


def mean_filter(I, ksize=5):
    I = cv2.blur(I, (ksize, ksize))
    return I
