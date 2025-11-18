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
    return np.array(src, copy=True)


def average_filter(src: np.ndarray, k_size: int) -> np.ndarray:
    """
    Moving average filter (aka box filter).

    You might want to use dip2.spatial_convolution(...) within this function.
    """
    # TO DO !!
    return np.array(src, copy=True)


def median_filter(src: np.ndarray, k_size: int) -> np.ndarray:
    """
    Median filter.
    """
    # TO DO !!
    return np.array(src, copy=True)


def bilateral_filter(src: np.ndarray, k_size: int, sigma_spatial: float, sigma_radiometric: float) -> np.ndarray:
    """
    Bilateral filter.
    """
    # TO DO !!
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
            return average_filter(src, 1)
        if noise_type is NoiseType.NOISE_TYPE_2:
            return average_filter(src, 1)
        raise ValueError("Unhandled noise type!")

    if noise_reduction_algorithm is NoiseReductionAlgorithm.NR_MEDIAN_FILTER:
        if noise_type is NoiseType.NOISE_TYPE_1:
            return median_filter(src, 1)
        if noise_type is NoiseType.NOISE_TYPE_2:
            return median_filter(src, 1)
        raise ValueError("Unhandled noise type!")

    if noise_reduction_algorithm is NoiseReductionAlgorithm.NR_BILATERAL_FILTER:
        if noise_type is NoiseType.NOISE_TYPE_1:
            return bilateral_filter(src, 1, 1.0, 1.0)
        if noise_type is NoiseType.NOISE_TYPE_2:
            return bilateral_filter(src, 1, 1.0, 1.0)
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
