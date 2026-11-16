"""
Classical Digital Image Processing (DIP) Pipeline for Knee Radiographs
========================================================================
In accordance with DIP principles and dip_project_upgrades.md:

Modules:
1. Mathematical ROI Joint Extraction (Sobel, Otsu, Morphology, Projection Profiling)
2. Frequency Domain Filtering (2D FFT, Butterworth Low-Pass, Homomorphic Filtering)
3. Handcrafted Texture Feature Extraction (GLCM Sclerosis & LBP Bone Margin Analysis)
4. Standardized Contrast Enhancement (CLAHE in Spatial Domain)
"""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


# =====================================================================
# 1. SPATIAL CONTRAST ENHANCEMENT (CLAHE)
# =====================================================================

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Applies Contrast Limited Adaptive Histogram Equalization in spatial domain.
    Prevents over-amplification of noise while sharpening joint space edges.
    """
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255.0).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(gray)
    return enhanced


# =====================================================================
# 2. MATHEMATICAL ROI JOINT LOCALIZATION (MORPHOLOGY & GRADIENTS)
# =====================================================================

def extract_knee_joint_roi(image, target_size=(224, 224), vertical_margin_ratio=0.35):
    """
    Algorithmic Region of Interest (ROI) extractor using classical DIP.
    Replaces black-box YOLO with mathematical edge detection, Otsu thresholding,
    and morphological opening/closing to isolate the articular joint space.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = np.copy(image)

    h, w = gray.shape

    # 1. Gradient computation using Sobel operators
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    grad_mag = np.uint8(255 * (grad_mag / (grad_mag.max() + 1e-8)))

    # 2. Otsu thresholding on gradient to isolate bone boundaries
    _, binary = cv2.threshold(grad_mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Morphological Operations:
    # Opening removes isolated speckle; Closing bridges trabecular gaps
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_large)

    # 4. Horizontal Projection Profile to detect articular space minimum
    # Bone density is highest above and below the joint; the joint gap appears as a valley
    horiz_profile = np.sum(closed[:, int(w * 0.2):int(w * 0.8)], axis=1)

    # Look for joint gap within middle 30% - 70% of the image height
    search_start = int(h * 0.3)
    search_end = int(h * 0.7)
    mid_profile = horiz_profile[search_start:search_end]

    if len(mid_profile) > 0:
        joint_y = search_start + int(np.argmin(mid_profile))
    else:
        joint_y = h // 2

    # Crop tightly around joint space
    crop_half_h = int(h * vertical_margin_ratio / 2.0)
    y1 = max(0, joint_y - crop_half_h)
    y2 = min(h, joint_y + crop_half_h)
    x1 = int(w * 0.05)
    x2 = int(w * 0.95)

    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        cropped = image

    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
    return resized


# =====================================================================
# 3. FREQUENCY DOMAIN FILTERING (2D FFT)
# =====================================================================

def butterworth_lowpass_filter(image, cutoff_freq=40, order=2):
    """
    2D Butterworth Low-Pass Filter in Frequency Domain.
    H(u,v) = 1 / (1 + (D(u,v) / D0)^(2n))
    Removes high-frequency quantum mottle without ringing artifacts.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2

    # 2D Fast Fourier Transform and shift low frequencies to center
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    # Compute Euclidean distance grid D(u,v)
    u = np.arange(rows) - crow
    v = np.arange(cols) - ccol
    U, V = np.meshgrid(u, v, indexing='ij')
    D = np.sqrt(U**2 + V**2)

    # Butterworth Transfer Function
    H = 1.0 / (1.0 + (D / (cutoff_freq + 1e-8))**(2 * order))

    # Apply filter and inverse transform
    fshift_filtered = fshift * H
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    return np.clip(img_back, 0, 255).astype(np.uint8)


def homomorphic_filter(image, cutoff=30, gamma_low=0.5, gamma_high=1.5, order=2):
    """
    Homomorphic Filtering in Frequency Domain.
    Separates illumination (low frequency) from reflectance (high frequency).
    Standardizes uneven X-ray beam exposure across bone margins.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Logarithmic transformation
    img_log = np.log1p(np.float64(gray))
    rows, cols = img_log.shape
    crow, ccol = rows // 2, cols // 2

    f = np.fft.fft2(img_log)
    fshift = np.fft.fftshift(f)

    u = np.arange(rows) - crow
    v = np.arange(cols) - ccol
    U, V = np.meshgrid(u, v, indexing='ij')
    D = np.sqrt(U**2 + V**2)

    # High-emphasis Butterworth transfer function
    H = (gamma_high - gamma_low) * (1.0 / (1.0 + (cutoff / (D + 1e-8))**(2 * order))) + gamma_low

    fshift_filtered = fshift * H
    img_back = np.fft.ifft2(np.fft.ifftshift(fshift_filtered))
    img_exp = np.expm1(np.real(img_back))

    # Normalize back to 0-255
    img_exp = cv2.normalize(img_exp, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(img_exp)


# =====================================================================
# 4. HANDCRAFTED TEXTURE FEATURE ANALYSIS (GLCM & LBP)
# =====================================================================

def extract_glcm_features(image, distances=[1, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """
    Gray-Level Co-occurrence Matrix (GLCM) Texture Descriptors.
    Quantifies Subchondral Sclerosis (increased bone density) and joint roughness:
      - Contrast: Local intensity variation.
      - Dissimilarity: Linear difference between neighboring pixels.
      - Homogeneity: Closeness of element distribution.
      - Energy: Uniformity / Angular Second Moment.
      - Correlation: Joint probability occurrence.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Quantize to 32 gray levels for stable statistical co-occurrence
    quantized = (gray // 8).astype(np.uint8)
    glcm = graycomatrix(quantized, distances=distances, angles=angles, levels=32, symmetric=True, normed=True)

    features = {
        "glcm_contrast": float(np.mean(graycoprops(glcm, "contrast"))),
        "glcm_dissimilarity": float(np.mean(graycoprops(glcm, "dissimilarity"))),
        "glcm_homogeneity": float(np.mean(graycoprops(glcm, "homogeneity"))),
        "glcm_energy": float(np.mean(graycoprops(glcm, "energy"))),
        "glcm_correlation": float(np.mean(graycoprops(glcm, "correlation"))),
    }
    return features


def extract_lbp_features(image, num_points=16, radius=2):
    """
    Local Binary Patterns (LBP) Histogram Descriptor.
    Captures rotational-invariant micro-textures of osteophytes (bone spurs).
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    lbp = local_binary_pattern(gray, P=num_points, R=radius, method="uniform")
    # Histogram of uniform patterns (num_points + 2 bins)
    hist, _ = np.histogram(lbp.ravel(), bins=num_points + 2, range=(0, num_points + 2), density=True)
    return hist.astype(np.float32)


# =====================================================================
# 5. FULL PIPELINE PREPROCESSOR
# =====================================================================

def preprocess_radiograph_full_dip(
    image,
    target_size=(224, 224),
    apply_roi_crop=False,
    use_butterworth=True,
    use_clahe_enhancement=True
):
    """
    End-to-End Classic DIP Preprocessing pipeline.
    Produces clean 3-channel input ready for the Deep Learning Attention Backbone.
    """
    processed = np.copy(image)
    if apply_roi_crop:
        processed = extract_knee_joint_roi(processed, target_size=target_size)

    if use_butterworth:
        processed = butterworth_lowpass_filter(processed, cutoff_freq=50, order=2)

    if use_clahe_enhancement:
        processed = apply_clahe(processed, clip_limit=2.0, tile_grid_size=(8, 8))

    if len(processed.shape) == 2:
        processed_rgb = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
    else:
        processed_rgb = processed

    if processed_rgb.shape[:2] != target_size:
        processed_rgb = cv2.resize(processed_rgb, target_size)

    # Normalize to [0.0, 1.0] float32
    return processed_rgb.astype(np.float32) / 255.0

