"""
DIP4: Image Restoration using Frequency Domain Filtering

This module implements image restoration techniques including:
- Discrete Fourier Transform (DFT) operations
- Inverse filtering
- Wiener filtering

Students should implement the functions marked with TODO.
"""

import numpy as np
import cv2


def degrade_image(img_in: np.ndarray, degraded_img: np.ndarray, kernel: np.ndarray, snr: float) -> tuple:
    """
    Degrade an image by convolution with a Gaussian kernel and adding Gaussian noise.
    This function is given - do not modify!
    
    Args:
        img_in: Input image (grayscale, float32)
        degraded_img: Output degraded image (will be modified)
        kernel: Output degradation kernel (will be modified)
        snr: Signal-to-noise ratio for added noise
        
    Returns:
        Tuple of (degraded_img, kernel)
    """
    kernel_size = 7
    
    # Create Gaussian kernel
    kernel = cv2.getGaussianKernel(kernel_size, kernel_size / 5.0, cv2.CV_32F)
    kernel = kernel @ kernel.T  # Outer product for 2D kernel
    
    # Pad kernel to image size
    kernel_padded = np.zeros(img_in.shape, dtype=np.float32)
    kernel_padded[:kernel_size, :kernel_size] = kernel
    
    # Circular shift to center
    kernel_padded = np.roll(kernel_padded, -(kernel_size // 2), axis=0)
    kernel_padded = np.roll(kernel_padded, -(kernel_size // 2), axis=1)
    
    # DFT of kernel
    kernel_freq = np.fft.fft2(kernel_padded)
    
    # DFT of input image
    img_freq = np.fft.fft2(img_in)
    
    # Multiply in frequency domain
    degraded_freq = img_freq * kernel_freq
    
    # Inverse DFT
    degraded_img = np.real(np.fft.ifft2(degraded_freq)).astype(np.float32)
    
    # Add noise
    noise_var = np.mean(degraded_img ** 2) / snr
    noise = np.random.randn(*degraded_img.shape).astype(np.float32) * np.sqrt(noise_var)
    degraded_img = degraded_img + noise
    
    return degraded_img, kernel_padded


def circ_shift(img_in: np.ndarray, shift_x: int, shift_y: int) -> np.ndarray:
    """
    Circular shift of an image.
    
    Args:
        img_in: Input image (can be real or complex)
        shift_x: Shift in x direction (columns)
        shift_y: Shift in y direction (rows)
        
    Returns:
        Shifted image
    """
    h, w = img_in.shape
    shifted = np.zeros_like(img_in)
    for i in range(h):
        for j in range(w):
            new_i = (i + shift_y) % h
            new_j = (j + shift_x) % w
            shifted[new_i, new_j] = img_in[i, j]

    return np.array(shifted, copy=True)


def dft_real2complex(img_in: np.ndarray) -> np.ndarray:
    """
    Perform DFT on a real-valued image and return complex result.
    
    Args:
        img_in: Input image (real-valued, float32)
        
    Returns:
        Complex DFT result
    """

    def dft_1d(source: np.ndarray) -> np.ndarray:
        N = source.size
        result = np.zeros(N, dtype=np.complex64)
        for k in range(N):
            k_N = k/N
            for n in range(N):
                result[k] += source[n] * np.exp(-2j * np.pi * k_N * n)
        return result

    def dft_2d(source: np.ndarray) -> np.ndarray:
        rows, cols = source.shape

        cols_dft = np.zeros((rows,cols), dtype=np.complex64)
        for c in range(cols):
            cols_dft[:, c] = dft_1d(source[:, c])

        full_dft = np.zeros((rows,cols), dtype=np.complex64)
        for r in range(rows):
            full_dft[r, :] = dft_1d(cols_dft[r, :])
        return full_dft

    # dft_complex = np.fft.fft2(img_in).astype(np.complex64) # numpy DFT
    dft_complex = dft_2d(img_in) # our original DFT
    return dft_complex



def idft_complex2real(img_in: np.ndarray) -> np.ndarray:
    """
    Perform inverse DFT on a complex image and return real result.
    
    Args:
        img_in: Complex input in frequency domain
        
    Returns:
        Real-valued spatial domain result (float32)
        
    """
    
    def idft_1d(freq: np.ndarray) -> np.ndarray:
        N = freq.size
        result = np.zeros(N, dtype=np.complex64)

        n = np.arange(N)
        for k in range(N):   
            result += freq[k] * np.exp(2j * np.pi * k * n / N)
        result /= N
        return result

    def idft_2d(freq2d: np.ndarray) -> np.ndarray:
        rows, cols = freq2d.shape

        cols_idft = np.zeros((rows, cols), dtype=np.complex64)
        for c in range(cols):
            cols_idft[:, c] = idft_1d(freq2d[:, c])

        full_idft = np.zeros((rows, cols), dtype=np.complex64)
        for r in range(rows):
            full_idft[r, :] = idft_1d(cols_idft[r, :])

        return full_idft

    # img_real =  np.fft.ifft2(img_in).real.astype(np.float32)
    idft_real = idft_2d(img_in).real.astype(np.float32)
    return idft_real
    


def apply_filter(img_in: np.ndarray, filter_spectrum: np.ndarray) -> np.ndarray:
    """
    Apply a filter in the frequency domain by element-wise multiplication.
    
    Args:
        img_in: Complex input spectrum
        filter_spectrum: Complex filter spectrum
        
    Returns:
        Filtered complex spectrum
        
    TODO: Implement this function
    """
    # TODO: Implement filter application (complex element-wise multiplication)
    return img_in * filter_spectrum
    #return img_in.copy()


def compute_inverse_filter(kernel: np.ndarray, eps: float) -> np.ndarray:
    """
    Compute inverse filter in frequency domain.
    
    Args:
        kernel: Degradation kernel in spatial domain
        eps: Threshold for small values (to avoid division by zero)
        
    Returns:
        Complex inverse filter spectrum
        
    """
    kernel = kernel.astype(np.float32)
    kernel_complex = dft_real2complex(kernel)
    
    magnitude = np.abs(kernel_complex)
    mask = magnitude >= eps
    
    inverse_filter = np.zeros_like(kernel_complex, dtype=np.complex64)
    inverse_filter[mask] = 1.0 / kernel_complex[mask]

    return inverse_filter


def inverse_filter(degraded: np.ndarray, kernel: np.ndarray, eps: float) -> np.ndarray:
    """
    Restore an image using inverse filtering.
    
    Args:
        degraded: Degraded input image
        kernel: Degradation kernel
        eps: Threshold for inverse filter computation
        
    Returns:
        Restored image (float32)
        
    TODO: Implement this function
    """
    # TODO: Implement inverse filtering
    G = dft_real2complex(degraded)
    H_inv = compute_inverse_filter(kernel,eps)
    F_hat = apply_filter(G,H_inv)
    restored = idft_complex2real(F_hat)

    return restored
    #return degraded.copy()


def compute_wiener_filter(kernel: np.ndarray, snr: float) -> np.ndarray:
    """
    Compute Wiener filter in frequency domain.
    
    Args:
        kernel: Degradation kernel in spatial domain
        snr: Signal-to-noise ratio estimate
        
    Returns:
        Complex Wiener filter spectrum
        
    TODO: Implement this function
    """
    # TODO: Implement Wiener filter computation
    kernel = kernel.astype(np.float32)


    kernel_shifted = circ_shift(kernel , -kernel.shape[0]//2, -kernel.shape[1]//2)
    # H = dft_real2complex(kernel_shifted)
    H = dft_real2complex(kernel)


    H_conj = np.conj(H)
    magnitude_squared = np.abs(H)**2
    W = H_conj/(magnitude_squared +1.0 / snr)

    return W.astype(np.complex64)
    #return np.ones(kernel.shape, dtype=np.complex64)



def wiener_filter(degraded: np.ndarray, kernel: np.ndarray, snr: float) -> np.ndarray:
    """
    Restore an image using Wiener filtering.
    
    Args:
        degraded: Degraded input image
        kernel: Degradation kernel
        snr: Signal-to-noise ratio estimate
        
    Returns:
        Restored image (float32)
        
    TODO: Implement this function
    """
    # TODO: Implement Wiener filtering
    G = dft_real2complex(degraded)
    W=compute_wiener_filter(kernel,snr)
    F_hat = apply_filter(G,W)
    restored = idft_complex2real(F_hat)
    return restored
    #return degraded.copy()
