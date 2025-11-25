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
    return np.zeros((1, k_size), dtype=np.float32)


def create_gaussian_kernel_2d(k_size: int) -> np.ndarray:
    """Generates 2D Gaussian filter kernel of given size."""
    # TO DO !!!

    # from bilateral
    # spatial kernel
    # distances = np.zeros((k_size,k_size), dtype=float)
    # kernel_center = int(k_size/2)
    # for y in range(k_size):
    #     for x in range(k_size):
    #         distances[y,x] = (x-kernel_center)**2 + (y-kernel_center)**2
    # spatial_kernel = (1/(2*np.pi*sigma_spatial**2))* np.exp(-distances/(2*sigma_spatial**2))
    return np.zeros((k_size, k_size), dtype=np.float32)


def circ_shift(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Perform a circular shift in (dx, dy) direction."""
    # TO DO !!!
    return np.array(image, copy=True)


def frequency_convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Performs convolution by multiplication in frequency domain."""
    # TO DO !!!
    return np.array(image, copy=True)


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
    return np.array(src, copy=True)


def usm(image: np.ndarray, filter_mode: FilterMode, size: int, thresh: float, scale: float) -> np.ndarray:
    """Performs unsharp masking to enhance image structures."""
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
