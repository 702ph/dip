#============================================================================
# Name        : dip2.py
# Author      : Annamalai Lakshmanan
# Version     : 1.0
# Copyright   : -
# Description : Core functionality stubs for second DIP assignment (Python port)
#============================================================================

from __future__ import annotations

from enum import Enum, auto
from typing import Dict

import numpy as np


class NoiseType(Enum):
    """Enumerates supported synthetic noise variants."""

    NOISE_TYPE_1 = auto()
    NOISE_TYPE_2 = auto()


noise_type_names: Dict[NoiseType, str] = {
    NoiseType.NOISE_TYPE_1: "NOISE_TYPE_1",
    NoiseType.NOISE_TYPE_2: "NOISE_TYPE_2",
}


class NoiseReductionAlgorithm(Enum):
    """Enumerates available denoising algorithms."""

    NR_MOVING_AVERAGE_FILTER = auto()
    NR_MEDIAN_FILTER = auto()
    NR_BILATERAL_FILTER = auto()


noise_reduction_algorithm_names: Dict[NoiseReductionAlgorithm, str] = {
    NoiseReductionAlgorithm.NR_MOVING_AVERAGE_FILTER: "NR_MOVING_AVERAGE_FILTER",
    NoiseReductionAlgorithm.NR_MEDIAN_FILTER: "NR_MEDIAN_FILTER",
    NoiseReductionAlgorithm.NR_BILATERAL_FILTER: "NR_BILATERAL_FILTER",
}


def spatial_convolution(src: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolution in spatial domain.

    Performs spatial convolution of image and filter kernel.
    """
    # TO DO !!
    flipkernel = np.flipud(np.fliplr(kernel))

    src_h, src_w = src.shape

    kernel_h, kernel_w = kernel.shape

    padding_height = kernel_h//2
    padding_width = kernel_w//2

    padded_image = np.pad(src,((padding_height,),(padding_width,)), "median")

    result = np.zeros_like(src, dtype=float)

    for y in range(src_h):
        for x in range (src_w):
            region = padded_image[y:y + kernel_h, x:x+kernel_w]
            result[y,x] = np.sum(region * flipkernel)

    return result
    return np.array(src, copy=True)


def average_filter(src: np.ndarray, k_size: int) -> np.ndarray:
    """
    Moving average filter (aka box filter).

    You might want to use dip2.spatial_convolution(...) within this function.
    """
    kernel = np.ones((k_size,k_size))
    kernel = np.multiply(kernel, (1/(k_size*k_size)))

    result = spatial_convolution(src, kernel)
    return result
    # TO DO !!
    return np.array(src, copy=True)


def median_filter(src: np.ndarray, k_size: int) -> np.ndarray:
    """
    Median filter.
    """
    padded_src = np.pad(src, ((k_size//2,),(k_size//2,)),"symmetric")
    src_h, src_w = src.shape
    result = np.zeros_like(src, dtype=float)
    for y in range(src_h):
        for x in range(src_w):
            region = padded_src[y:y+k_size , x:x+k_size]
            if x == 5 and y == 5:
                print(region.flatten())
            #region = sorted(region)
            flat_region = region.flatten()
            sorted_region = np.array(sorted((flat_region)))
            if x == 5 and y == 5:
                print('after sort: ',sorted_region)
            result[y,x] = np.median(sorted_region)
            
    # TO DO !!
    return result
    return np.array(src, copy=True)


def bilateral_filter(src: np.ndarray, k_size: int, sigma_spatial: float, sigma_radiometric: float) -> np.ndarray:
    """
    Bilateral filter.
    """
    padded_image = np.pad(src, ((k_size//2,),(k_size//2,)),"median")

    result = np.zeros_like(src, dtype=float)
    img_h, img_w = src.shape

    spatialmat = np.zeros((k_size,k_size))
    center = [k_size//2, k_size//2]
    for y in range(k_size):
        for x in range(k_size):
            weight = (1/(2*np.pi*np.square(sigma_spatial)))*np.exp(- (np.square(y-center[0])+np.square(x-center[1])) / (2*np.square(sigma_spatial) ))
            spatialmat[y,x] = weight
    
    #print(spatialmat)

    
    for y in range (img_h):
        for x in range (img_w):
            r_w = np.zeros_like(spatialmat)
            area = padded_image[y:y+k_size, x:x+k_size]
            for ky in range(k_size):
                for kx in range(k_size):
                    weightrad = 1 / (2*np.pi*sigma_radiometric**2)*np.exp(
                        - (((area[ky,kx]-area[center[0],center[1]])**2) / (2*sigma_radiometric**2))
                    )
                    r_w[ky,kx] = weightrad
            final_kernel = np.multiply(spatialmat, r_w)
            final_kernel /= np.sum(final_kernel)
            result[y,x] = np.sum(area * final_kernel)            
    # TO DO !!
    return result
    return np.array(src, copy=True)


def nlm_filter(src: np.ndarray, search_size: int, sigma: float) -> np.ndarray:
    """
    Non-local means filter (optional task!).
    """
    return np.array(src, copy=True)


def choose_best_algorithm(noise_type: NoiseType) -> NoiseReductionAlgorithm:
    """
    Chooses the right algorithm for the given noise type.
    """
    # TO DO !!
    match noise_type:
        case NoiseType.NOISE_TYPE_1:
            return NoiseReductionAlgorithm.NR_MEDIAN_FILTER
        case NoiseType.NOISE_TYPE_2:
            return NoiseReductionAlgorithm.NR_BILATERAL_FILTER
        case _:
            return NoiseReductionAlgorithm.NR_MOVING_AVERAGE_FILTER

    raise NotImplementedError("Student implementation missing")


def denoise_image(
    src: np.ndarray,
    noise_type: NoiseType,
    noise_reduction_algorithm: NoiseReductionAlgorithm,
) -> np.ndarray:
    """
    Denoising, with parameters specifically tweaked to the supported noise types.
    """
    # TO DO !!

    if noise_reduction_algorithm is NoiseReductionAlgorithm.NR_MOVING_AVERAGE_FILTER:
        if noise_type is NoiseType.NOISE_TYPE_1:
            return average_filter(src, 5)
        if noise_type is NoiseType.NOISE_TYPE_2:
            return average_filter(src, 3)
        raise ValueError("Unhandled noise type!")

    if noise_reduction_algorithm is NoiseReductionAlgorithm.NR_MEDIAN_FILTER:
        if noise_type is NoiseType.NOISE_TYPE_1:
            return median_filter(src, 5)
        if noise_type is NoiseType.NOISE_TYPE_2:
            return median_filter(src, 5)
        raise ValueError("Unhandled noise type!")

    if noise_reduction_algorithm is NoiseReductionAlgorithm.NR_BILATERAL_FILTER:
        if noise_type is NoiseType.NOISE_TYPE_1:
            return bilateral_filter(src, 9, 2, 100.0)
        if noise_type is NoiseType.NOISE_TYPE_2:
            return bilateral_filter(src, 9, 2, 100.0)
        raise ValueError("Unhandled noise type!")

    raise ValueError("Unhandled filter type!")


__all__ = [
    "NoiseType",
    "NoiseReductionAlgorithm",
    "noise_type_names",
    "noise_reduction_algorithm_names",
    "spatial_convolution",
    "average_filter",
    "median_filter",
    "bilateral_filter",
    "nlm_filter",
    "choose_best_algorithm",
    "denoise_image",
]
