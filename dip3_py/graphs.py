import sys
from enum import Enum, auto
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
import cv2


# Z axis: time (idx)
# Y axis: number of pixels in input image
# X axis: filter (kernel) size

import time


def circ_shift(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    start = time.time()
    """Perform a circular shift in (dx, dy) direction."""
    # TO DO !!!
    """Perform a circular shift in (dx, dy) direction."""
    print('image shape: ', image.shape[0])
    h = image.shape[0]
    w = image.shape[0]

    shifted = np.zeros_like(image)
    for i in range(h):
        for j in range(w):
            new_i = (i + dy) % h
            new_j = (j + dx) % w
            shifted[new_i, new_j] = image[i, j]
    end = time.time()
    elapsed_time = end-start

    return np.array(shifted, copy=True)



def frequency_convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    start = time.time()
    """Performs convolution by multiplication in frequency domain."""
    # TO DO !!!
    kernel_h, kernel_w = kernel.shape
    #image_w, image_h = image.shape
    #print('kernel: ', kernel)
    print(-2 % 5)
    # in exercise slide: "Copy the kernel into a larger matrix"
    padded_kernel = np.zeros_like(image, dtype=np.float32)
    dx = -int(kernel_w / 2)
    dy = -int(kernel_h / 2)
    #dx = -image_w // 2
    #dy = -image_h // 2
    
    padded_kernel[0 : kernel_h, 0:kernel_w] = kernel

    print('len padded: ',len(padded_kernel))
    print('dx: ', dx)
    print('kernel_w: ', kernel_w)
    shifted_kernel = circ_shift(padded_kernel, dx, dy)
    #print('shifted_kernel: ', shifted_kernel)

    #Fourier Transform: from spatial to frequency domain
    dft_image = cv2.dft(image, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_kernel = cv2.dft(shifted_kernel, flags=cv2.DFT_COMPLEX_OUTPUT)

    # convolution
    dft_result = cv2.mulSpectrums(dft_image, dft_kernel, 0)

    #Inverse Fourier
    result = cv2.idft(dft_result, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    end = time.time()
    elapsed_time = end-start
    # return np.array(result, copy=True)
    return np.clip(result, 0, 255) , elapsed_time



def separable_filter(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    start = time.time()
    """Convolution in spatial domain by separable filters."""
    # TO DO !!!
    # filtering in x
    horizontal = spatial_convolution(image, kernel)

    #filtering in y
    kernel_transposed = kernel.T
    result = spatial_convolution(horizontal, kernel_transposed)
    end = time.time()
    elapsed_time = end-start
    return np.array(result, copy=True) , elapsed_time


def sat_filter(image: np.ndarray, size: int) -> np.ndarray:
    """Convolution in spatial domain using integral images."""
    # TO DO !!!
    return np.array(image, copy=True)



def spatial_convolution(src: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    start = time.time()
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
    end = time.time()
    elapsed_time = end-start
    return np.array(result, copy=True) , elapsed_time


# Small test
size = 5
image = np.ones((size,size))

kernel_size = 2
kernel = np.ones((kernel_size, kernel_size))


# Loop example
filter_size = np.arange(1,10,1)
image_size = np.arange(1,10,1)
#print(image_size)
time_spatial = []

for k in filter_size:
    for j in image_size:
        print('filter: ', k)
        print('image: ', j)
        image = np.ones((j,j))
        num_pixels = image_size**2
        kernel = np.ones((k, k))
        freq_conv_spatial, elapsed_time_spatial = spatial_convolution(image, kernel)
        time_spatial.append(elapsed_time_spatial)
        
time_spatial = np.array(time_spatial)

fig = plt.figure()
ax = plt.axes(projection='3d')


ax.plot_trisurf(filter_size, image_size, time_spatial)
ax.set_title('Time Data for Spatial Convolution')
ax.set_xlabel('Filter Size')
ax.set_ylabel('Number of pixels in input image')
ax.view_init(-120, 80)
plt.show()

print('spatial time: ',elapsed_time_spatial)





freq_conv_seperable, elapsed_time_seperable = spatial_convolution(image, kernel)
print('seperable time: ',elapsed_time_seperable)




freq_conv, elapsed_time_freq = frequency_convolution(image, kernel)
print('freq time: ',elapsed_time_freq)


