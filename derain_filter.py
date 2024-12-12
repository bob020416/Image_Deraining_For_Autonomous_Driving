import cv2
import numpy as np


def derain_filter(I, opt, iterations=5, verbose=False):
    """
    Main function to perform image deraining with iterative refinement.

    Parameters:
    - I: Input image as a NumPy array (H x W x 3).
    - opt: Option parameter (0 or 1) for blending factor.
    - iterations: Number of iterations for refinement.
    """
    I_input = I.copy().astype(np.float32)
    for i in range(iterations):
        # Low-frequency analysis
        if verbose:
            print(f"[{i+1}-1] Performing low-frequency analysis...")
        Il = lf_analysis(I_input)
        # Apply guided filter to low-frequency component
        if verbose:
            print(f"[{i+1}-2] Applying guided filter to low-frequency component...")
        Il = guided_filter(Il, Il, radius=8, eps=0.01)
        # High-frequency component
        if verbose:
            print(f"[{i+1}-3] Computing high-frequency component...")
        Ih = I_input - Il.astype(np.float32)
        # High-frequency analysis
        if verbose:
            print(f"[{i+1}-4] Performing high-frequency analysis...")
        Ih_final = hf_analysis(Il, Ih, I, opt)
        # Update input for next iteration
        if verbose:
            print(f"[{i+1}-5] Updating input for next iteration...")
        I_input = Ih_final.astype(np.float32)

    return Ih_final


def lf_analysis(I):
    """
    Low-frequency analysis: Detects rain/snow pixels and fills holes.

    Parameters:
    - I: Input image as a NumPy array (H x W x 3).

    Returns:
    - Il: Low-frequency component of the image (H x W x 3).
    """
    Mi = detect_rain_snow(I)
    Iha = I.astype(np.float32) * Mi  # Hadamard product
    Im = np.zeros_like(Iha)

    # Process each channel independently
    for c in range(3):
        Im[:, :, c] = fill_hole_mean(Iha[:, :, c], Mi[:, :, c])

    Il = Im.astype(np.uint8)
    return Il


def hf_analysis(Il, Ih, I, opt, verbose=False):
    """
    High-frequency analysis: Enhances edges and reconstructs the final image.

    Parameters:
    - Il: Low-frequency component (H x W x 3).
    - Ih: High-frequency component (H x W x 3).
    - I: Original image (H x W x 3).
    - opt: Option parameter for blending.

    Returns:
    - Ih_final: Final reconstructed image after high-frequency analysis (H x W x 3).
    """
    # Layer 1 operation: Edge enhancement
    Il_edge = Il.copy().astype(np.float32)
    laplacian_kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
    Il_edge += 0.1 * cv2.filter2D(Il_edge, -1, laplacian_kernel)
    Il_edge = np.clip(Il_edge, 0, 255).astype(np.uint8)
    if verbose:
        cv2.imwrite("output/edge_enhanced.jpg", Il_edge)

    # Guided filtering
    Ih_new = guided_filter(Il_edge, Ih.astype(np.uint8), radius=8, eps=0.01)
    Ir = Ih_new.astype(np.float32) + Il.astype(np.float32)
    Ir = np.clip(Ir, 0, 255).astype(np.uint8)
    if verbose:
        cv2.imwrite("output/recovered.jpg", Ir)

    # Layer 2 operation
    Icr = np.minimum(Ir, I.astype(np.uint8))
    if verbose:
        cv2.imwrite("output/clear_recovered.jpg", Icr)

    # Layer 3 operation
    b = 0.9 if opt == 0 else 0.5
    Iref = (b * Icr.astype(np.float32)) + ((1 - b) * Ir.astype(np.float32))
    Iref = np.clip(Iref, 0, 255).astype(np.uint8)

    # Guided filtering
    Irr = guided_filter(Icr, Iref, radius=8, eps=0.01)
    Ih_final = np.clip(Irr, 0, 255).astype(np.uint8)

    return Ih_final


def detect_rain_snow(I):
    """
    Detects rain/snow pixels in the image by analyzing each color channel.

    Parameters:
    - I: Input image as a NumPy array (H x W x 3).

    Returns:
    - Mi: Mask indicating rain/snow pixels (H x W x 3).
    """
    # Split the image into R, G, B channels
    imgR, imgG, imgB = cv2.split(I.astype(np.uint8))

    # Get Mi for each channel
    MiR = getMi(imgR)
    MiG = getMi(imgG)
    MiB = getMi(imgB)

    # Combine Mi from all channels: set to 0 only if all channels indicate rain
    rain_mask = (MiR == 0) & (MiG == 0) & (MiB == 0)

    # Initialize Mi with ones
    Mi = np.ones_like(imgR, dtype=np.float32)
    Mi[rain_mask] = 0

    # Stack Mi into three channels
    Mi = np.stack([Mi, Mi, Mi], axis=2)

    return Mi


def getMi(I_channel):
    """
    Computes the Mi mask for a single color channel.

    Parameters:
    - I_channel: Single color channel image (H x W).

    Returns:
    - Mi: Mask for the channel (H x W).
    """
    # Pad the image with 6 pixels on all sides using edge values
    I_repl = np.pad(I_channel, 6, mode="edge")

    # Compute local means
    Imid = cv2.blur(I_repl, (7, 7))[6:-6, 6:-6]
    ITL = cv2.blur(I_repl[6:, 6:], (7, 7))[:-6, :-6]
    ITR = cv2.blur(I_repl[6:, :-6], (7, 7))[:-6, 6:]
    IBL = cv2.blur(I_repl[:-6, 6:], (7, 7))[6:, :-6]
    IBR = cv2.blur(I_repl[:-6, :-6], (7, 7))[6:, 6:]

    # Compute Imax
    Imax = np.maximum.reduce([Imid, ITL, ITR, IBL, IBR])

    # Detect rain/snow pixels
    Mi = np.ones_like(I_channel, dtype=np.float32)
    Mi[I_channel > Imax] = 0
    return Mi


def fill_hole_mean(I, Mi):
    """
    Fills the detected rain/snow pixels with the maximum local mean from neighboring regions.

    Parameters:
    - I: Single color channel image with rain pixels masked (H x W).
    - Mi: Mask indicating rain pixels (H x W).

    Returns:
    - Im: Image with rain pixels filled (H x W).
    """
    # Pad the image with 6 pixels on all sides using edge values
    I_repl = np.pad(I, 6, mode="edge")

    # Compute local means
    Imid = cv2.blur(I_repl, (7, 7))[6:-6, 6:-6]
    ITL = cv2.blur(I_repl[6:, 6:], (7, 7))[:-6, :-6]
    ITR = cv2.blur(I_repl[6:, :-6], (7, 7))[:-6, 6:]
    IBL = cv2.blur(I_repl[:-6, 6:], (7, 7))[6:, :-6]
    IBR = cv2.blur(I_repl[:-6, :-6], (7, 7))[6:, 6:]

    # Compute Imax
    Imax = np.maximum.reduce([Imid, ITL, ITR, IBL, IBR])

    # Initialize output image
    Im = I.copy()

    # Create mask of pixels to fill
    mask = Mi == 0

    # Fill holes with Imax
    Im[mask] = Imax[mask]

    return Im


def guided_filter(guide, src, radius=8, eps=0.01):
    """
    Applies guided filtering to the source image using the guide image.

    Parameters:
    - guide: Guide image (H x W) or (H x W x 3).
    - src: Source image to be filtered (H x W) or (H x W x 3).
    - radius: Radius of the guided filter.
    - eps: Regularization parameter.

    Returns:
    - q: Filtered image.
    """
    # Check if guide is single-channel or multi-channel
    if len(guide.shape) == 2:
        guide = guide[..., np.newaxis]
    if len(src.shape) == 2:
        src = src[..., np.newaxis]

    # Initialize output
    q = np.zeros_like(src, dtype=np.float32)

    # Apply guided filter to each channel
    for c in range(src.shape[2]):
        q[:, :, c] = cv2.ximgproc.guidedFilter(
            guide=guide[:, :, c].astype(np.uint8),
            src=src[:, :, c].astype(np.uint8),
            radius=radius,
            eps=eps,
        )
    return q
