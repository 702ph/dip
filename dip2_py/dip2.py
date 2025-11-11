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
    # NR_BILATERAL_FILTER = auto()
    NR_NLM_FILTER = auto()



noise_reduction_algorithm_names: Dict[NoiseReductionAlgorithm, str] = {
    NoiseReductionAlgorithm.NR_MEDIAN_FILTER: "NR_MEDIAN_FILTER",
    NoiseReductionAlgorithm.NR_MOVING_AVERAGE_FILTER: "NR_MOVING_AVERAGE_FILTER",
    # NoiseReductionAlgorithm.NR_BILATERAL_FILTER: "NR_BILATERAL_FILTER",
    NoiseReductionAlgorithm.NR_NLM_FILTER: "NR_NLM_FILTER",
}


def spatial_convolution(src: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolution in spatial domain.
    Performs spatial convolution of image and filter kernel.
    """
    # TO DO !!
    # result = ((src * np.flip(kernel)).sum()/kernel.size).astype(np.uint8)
    result = ((src * np.flip(kernel)).sum()).astype(np.uint8)
    return np.array(result, copy=True)


def average_filter(src: np.ndarray, k_size: int) -> np.ndarray:
    """
    Moving average filter (aka box filter).
    You might want to use dip2.spatial_convolution(...) within this function.
    """
    # TO DO !!
    print("debug")
    pad_size = int(k_size/2)
    padded_src = np.pad(src, (pad_size, pad_size), mode="reflect")

    height = src.shape[0]
    width = src.shape[1]
    filtered_img = np.empty((height, width), dtype=np.uint8)

    kernel = np.ones((k_size, k_size))
    kernel /= kernel.sum() #normalization

    for y in range(height):
        for x in range(width):
            crapped = padded_src[y:y+k_size, x:x+k_size]
            filtered_img[y,x] = spatial_convolution(crapped, kernel)

    return np.array(filtered_img, copy=True)



def median_filter(src: np.ndarray, k_size: int) -> np.ndarray:
    """
    Median filter.
    """
    # TO DO !!
    pad_size = int(k_size/2)
    padded_src = np.pad(src, (pad_size, pad_size), mode="reflect")

    height = src.shape[0]
    width = src.shape[1]
    filtered_img = np.empty((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            crapped = padded_src[y:y+k_size, x:x+k_size]
            filtered_img[y,x] = np.median(crapped).astype(np.uint8)

    return np.array(filtered_img, copy=True)



def bilateral_filter(src: np.ndarray, k_size: int, sigma_spatial: float, sigma_radiometric: float) -> np.ndarray:
    """
    Bilateral filter.
    """
    # TO DO !!
    print("bilateral_filter():")
    pad_size = int(k_size/2)
    padded_src = np.pad(src, (pad_size, pad_size), mode="reflect")

    height = src.shape[0]
    width = src.shape[1]
    filtered_img = np.empty((height, width), dtype=np.uint8)

    # spatial kernel
    distances = np.zeros((k_size,k_size), dtype=float)
    kernel_center = int(k_size/2)
    for y in range(k_size):
        for x in range(k_size):
            distances[y,x] = (x-kernel_center)**2 + (y-kernel_center)**2
    spatial_kernel = np.exp(-distances/(2*sigma_spatial**2))

    # iterate over image
    for y in range(height):
        for x in range(width):
            crapped = padded_src[y:y+k_size, x:x+k_size]

            # radiometric kernel
            center_pix = crapped[kernel_center, kernel_center]
            diff_intensity = (crapped - center_pix)**2
            radiometric_kernel = np.exp(-diff_intensity/(2*sigma_radiometric**2))

            # combine kernels
            kernel = spatial_kernel*radiometric_kernel
            kernel /= kernel.sum() #normalization
            filtered_img[y,x] = spatial_convolution(crapped, kernel)

    return np.array(filtered_img, copy=True)



def nlm_filter(src: np.ndarray, search_size: int, sigma: float) -> np.ndarray:
    """
    Non-local means filter (optional task!).
    """
    print("nlm_filter():")
    patch_size = 3

    pad_size = int(search_size/2)
    padded_src = np.pad(src, (pad_size, pad_size), mode="reflect")

    height = src.shape[0]
    width = src.shape[1]
    filtered_img = np.zeros((height, width), dtype=np.uint8)

    denom = 2.0 * (sigma * sigma)
    eps = 1e-8 #to avoid 0

    # iterate over image
    search_window_pad_size = int(patch_size/2)
    for y in range(height):
        for x in range(width):

            # crop search_window
            search_window = padded_src[0:y+search_size, 0:x+search_size]
            padded_search_window = np.pad(search_window, (search_window_pad_size, search_window_pad_size), mode="reflect")

            # center path
            center_patch = padded_search_window[y:(y+patch_size), x:(x+patch_size)]
            center_mean = center_patch.mean()

            weight_sum = 0.0
            weighted_pixel_sum = 0.0
            for y_s in range(search_size):
                for x_s in range(search_size):
                    patch = padded_search_window[y_s:(y_s+patch_size), x_s:(x_s+patch_size)]
                    path_mean = patch.mean()
                    distance = (path_mean - center_mean)**2
                    weight = np.exp(-distance/ denom)
                    weight_sum += weight
                    weighted_pixel_sum += weight * padded_search_window[y_s, x_s]

            filtered_img[y,x] = weighted_pixel_sum / (weight_sum + eps)
            # filtered_img[y,x] = weighted_pixel_sum / (weight_sum)

    return np.array(filtered_img, copy=True)


def choose_best_algorithm(noise_type: NoiseType) -> NoiseReductionAlgorithm:
    """
    Chooses the right algorithm for the given noise type.
    """
    # TO DO !!
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
            return average_filter(src, 3)
        if noise_type is NoiseType.NOISE_TYPE_2:
            return average_filter(src, 3)
        raise ValueError("Unhandled noise type!")

    if noise_reduction_algorithm is NoiseReductionAlgorithm.NR_MEDIAN_FILTER:
        if noise_type is NoiseType.NOISE_TYPE_1:
            return median_filter(src, 5)
        if noise_type is NoiseType.NOISE_TYPE_2:
            return median_filter(src, 5)
        raise ValueError("Unhandled noise type!")

    # if noise_reduction_algorithm is NoiseReductionAlgorithm.NR_BILATERAL_FILTER:
    #     if noise_type is NoiseType.NOISE_TYPE_1:
    #         return bilateral_filter(src, 3, 1.0, 1.0)
    #     if noise_type is NoiseType.NOISE_TYPE_2:
    #         return bilateral_filter(src, 3, 1.0, 1.0)
    #     raise ValueError("Unhandled noise type!")

    if noise_reduction_algorithm is NoiseReductionAlgorithm.NR_NLM_FILTER:
        if noise_type is NoiseType.NOISE_TYPE_1:
            return nlm_filter(src, 5, 1.0)
        if noise_type is NoiseType.NOISE_TYPE_2:
            return nlm_filter(src, 5, 1.0)
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
