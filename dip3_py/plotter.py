from dip3 import spatial_convolution, frequency_convolution, separable_filter
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# =============================
# Parameters
# =============================
image_sizes = np.arange(10, 501, 10)    # 10,20,...,500
kernel_sizes = np.arange(10, 101, 10)     # 3,5,7,...,99   (not larger than smallest image)

# Time-matrices (image_size × kernel_size)
T_spatial   = np.zeros((len(image_sizes), len(kernel_sizes)))
T_frequency = np.zeros((len(image_sizes), len(kernel_sizes)))
T_separable = np.zeros((len(image_sizes), len(kernel_sizes)))



# =============================
# Measurement loops
# =============================
for i, imsize in enumerate(image_sizes):
    img = np.random.randint(0, 256, (imsize, imsize), dtype=np.uint8).astype(np.float32)

    for j, ksize in enumerate(kernel_sizes):

        # Kernel must be <= image size
        if ksize > imsize:
            continue

        # Generate random floating-point kernel
        kernel = np.random.rand(ksize, ksize).astype(np.float32)

        # ------------- spatial convolution -------------
        start = time.time()
        spatial_convolution(img, kernel)
        T_spatial[i, j] = time.time() - start

        # ------------- frequency convolution -------------
        start = time.time()
        frequency_convolution(img, kernel)
        T_frequency[i, j] = time.time() - start

        # ------------- separable filter (if kernel is separable) -------------
        # For benchmarking: just call function anyway
        start = time.time()
        separable_filter(img, kernel_for_seperable)
        T_separable[i, j] = time.time() - start
        
        print('j: ',j)
    print('i: ',i)


# =============================
# 3D Surface Plot Function
# =============================
def plot_surface(Z, title):


    X, Y = np.meshgrid(kernel_sizes, image_sizes)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel("Kernel size")
    ax.set_ylabel("Image size")
    ax.set_zlabel("Time (seconds)")

     # Set Y-axis ticks at original image_sizes positions
    yticks = ax.get_yticks()  # get default tick positions
    # Only keep ticks that match image_sizes
    yticks = [y for y in yticks if y >= image_sizes[0] and y <= image_sizes[-1]]
    ax.set_yticks(yticks)
    # Set the labels to be squared values
    yticklabels = [int(y**2) for y in yticks]
    ax.set_yticklabels(yticklabels)
    plt.savefig(f'{title}.png')
    plt.show()
    plt.pause(1)


# =============================
# Plot all three surfaces
# =============================


plot_surface(T_spatial,   "Spatial Convolution Time")
#plt.savefig(spatial_plot, 'Spatial_Convolution.png')
plot_surface(T_frequency, "Frequency-Domain Convolution Time")

plot_surface(T_separable, "Separable Filter Time")
