#============================================================================
# Name        : dip3.py
# Author      : Annamalai Lakshmanan
# Version     : 1.0
# Description : Core functionality stubs for third DIP assignment (Python port)
#============================================================================

from __future__ import annotations

from enum import Enum, auto
from typing import Dict

import numpy as np
import cv2

class FilterMode(Enum):
    """Supported convolution backends."""

    FM_SPATIAL_CONVOLUTION = auto()
    FM_FREQUENCY_CONVOLUTION = auto()
    FM_SEPERABLE_FILTER = auto()
    FM_INTEGRAL_IMAGE = auto()


filter_mode_names: Dict[FilterMode, str] = {
    FilterMode.FM_SPATIAL_CONVOLUTION: "FM_SPATIAL_CONVOLUTION",
    FilterMode.FM_FREQUENCY_CONVOLUTION: "FM_FREQUENCY_CONVOLUTION",
    FilterMode.FM_SEPERABLE_FILTER: "FM_SEPERABLE_FILTER",
    FilterMode.FM_INTEGRAL_IMAGE: "FM_INTEGRAL_IMAGE",
}


def create_gaussian_kernel_1d(k_size: int) -> np.ndarray:
    """Generates 1D Gaussian filter kernel of given size."""
    # TO DO !!!
    mu = int(k_size/2) #kernel center
    sigma = int(k_size/5) # taken from the slide

    coordinates = np.arange(-mu, mu + 1)
    distances_sq = coordinates ** 2
    # e.g.
    # np.arange(-mu, mu + 1)
    # array([-2, -1, 0, 1, 2])

    spatial_kernel =  np.exp(-distances_sq / (2 * sigma ** 2))
    spatial_kernel /= np.sum(spatial_kernel) # to ensure sum to exactly 1.0
    spatial_kernel = spatial_kernel.reshape(1, -1) # test expect a shape of (1, N)
    return spatial_kernel



def create_gaussian_kernel_2d(k_size: int) -> np.ndarray:
    """Generates 2D Gaussian filter kernel of given size."""
    # TO DO !!!

    mu = int(k_size/2) #kernel center
    sigma = int(k_size/5) # taken from the slide

    # spatial kernel from bilateral filter implementation
    # for y in range(k_size):
    #     for x in range(k_size):
    #         distances[y,x] = (x-mu)**2 + (y-mu)**2
    # spatial_kernel = (1/(2*np.pi*sigma**2))* np.exp(-distances/(2*sigma**2))


    # improved implementation
    y, x = np.ogrid[-mu:mu + 1, -mu:mu + 1]
    distances_sq = x ** 2 + y ** 2
    #e.g. mu = 2
    # y, x = np.ogrid[-mu:mu + 1, -mu:mu + 1]
    # y
    # array([[-2],
    #        [-1],
    #        [0],
    #        [1],
    #        [2]])
    # x
    # array([[-2, -1, 0, 1, 2]])
    # distances_sq = x ** 2 + y ** 2

    # distances_sq
    # array([[8, 5, 4, 5, 8],
    #        [5, 2, 1, 2, 5],
    #        [4, 1, 0, 1, 4],
    #        [5, 2, 1, 2, 5],
    #        [8, 5, 4, 5, 8]])

    # kernel with (1.0 / (2 * np.pi * sigma ** 2)) part
    # spatial_kernel = (1.0 / (2 * np.pi * sigma ** 2)) * np.exp(-distances_sq / (2 * sigma ** 2))

    # kernel normalization at the end
    spatial_kernel =  np.exp(-distances_sq / (2 * sigma ** 2))
    spatial_kernel /= np.sum(spatial_kernel) # to ensure sum to exactly 1.0

    return spatial_kernel



def circ_shift(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Perform a circular shift in (dx, dy) direction."""
    # TO DO !!!
    """Perform a circular shift in (dx, dy) direction."""
    h, w = image.shape
    shifted = np.zeros_like(image)
    for i in range(h):
        for j in range(w):
            new_i = (i + dy) % h
            new_j = (j + dx) % w
            shifted[new_i, new_j] = image[i, j]
    return np.array(shifted, copy=True)



def frequency_convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Performs convolution by multiplication in frequency domain."""
    # TO DO !!!
    kernel_h, kernel_w = kernel.shape

    # in exercise slide: "Copy the kernel into a larger matrix"
    padded_kernel = np.zeros_like(image, dtype=np.float32)
    dx = -kernel_w // 2
    dy = -kernel_h // 2
    padded_kernel[0 : kernel_h, 0:kernel_h] = kernel
    shifted_kernel = circ_shift(kernel, dx, dy)

    #Fourier Transform: from spatial to frequency domain
    dft_image = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_kernel = cv2.dft(shifted_kernel, flags=cv2.DFT_COMPLEX_OUTPUT)

    # convolution
    dft_result = cv2.mulSpectrums(dft_image, dft_kernel, 0)

    #Inverse Fourier
    result = cv2.idft(dft_result, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

    return np.array(result, copy=True)



def separable_filter(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolution in spatial domain by separable filters."""
    # TO DO !!!
    return np.array(image, copy=True)


def sat_filter(image: np.ndarray, size: int) -> np.ndarray:
    """Convolution in spatial domain using integral images."""
    # TO DO !!!
    return np.array(image, copy=True)


def spatial_convolution(src: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolution in spatial domain."""
    # Hopefully already DONE, copy from last homework
    # TO DO !!
    flipkernel = np.flipud(np.fliplr(kernel))

    src_h, src_w = src.shape
    kernel_h, kernel_w = kernel.shape
    padding_height = kernel_h // 2
    padding_width = kernel_w // 2

    padded_image = np.pad(src, ((padding_height,), (padding_width,)), "reflect")
    result = np.zeros_like(src, dtype=float)

    for y in range(src_h):
        for x in range(src_w):
            region = padded_image[y:y + kernel_h, x:x + kernel_w]
            result[y, x] = np.sum(region * flipkernel)
    # return result
    return np.array(result, copy=True)


def usm(image: np.ndarray, filter_mode: FilterMode, size: int, thresh: float, scale: float) -> np.ndarray:
    """Performs unsharp masking to enhance image structures."""
    """ size: kernel size"""
    # TO DO !!!
    # use smooth_image(...) for smoothing
    return np.array(image, copy=True)


def smooth_image(image: np.ndarray, size: int, filter_mode: FilterMode) -> np.ndarray:
    """Performs smoothing operation choosing the algorithm by filter_mode."""
    if filter_mode is FilterMode.FM_SPATIAL_CONVOLUTION:
        return spatial_convolution(image, create_gaussian_kernel_2d(size))
    if filter_mode is FilterMode.FM_FREQUENCY_CONVOLUTION:
        return frequency_convolution(image, create_gaussian_kernel_2d(size))
    if filter_mode is FilterMode.FM_SEPERABLE_FILTER:
        return separable_filter(image, create_gaussian_kernel_1d(size))
    if filter_mode is FilterMode.FM_INTEGRAL_IMAGE:
        return sat_filter(image, size)
    raise ValueError("Unhandled filter type!")


__all__ = [
    "FilterMode",
    "filter_mode_names",
    "create_gaussian_kernel_1d",
    "create_gaussian_kernel_2d",
    "circ_shift",
    "frequency_convolution",
    "separable_filter",
    "sat_filter",
    "spatial_convolution",
    "usm",
    "smooth_image",
]
