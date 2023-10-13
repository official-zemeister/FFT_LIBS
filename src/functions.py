import numpy as np


def get_freqs(mmap, mmap_step):
    """
    Input:
        mmap: array_like
            Libs map data. Shape should be (x_size, y_size, wavelengths)
        mmap_set: int or float
            Resolution of the LIBS map
    Returns:
        freqs_x: array_like
        freqs_y: array_like
            Returns the k_space components along each transverse axis of mmap
    """
    nx, ny = mmap.shape[1], mmap.shape[0]
    dx, dy = mmap_step, mmap_step

    freqs_x = 2*np.pi*np.fft.fftfreq(nx, dx)
    freqs_y = 2*np.pi*np.fft.fftfreq(ny, dy)
    
    return freqs_x, freqs_y

def size_relation(size, mmap_step):
    """
    Input:
        size: int or float
            Size in pixels of the objects to filter
        mmap_step: int or float
            Resolution of the LIBS map
    Returns:
        size_kspace: float
            Returns the k_space frequency corresponding to the object size
    """
    object_size = size*mmap_step
    size_kspace = 2*np.pi/object_size/2

    return size_kspace

def dist(xx, yy):
    """
    Input:
        xx: float or array_like
            Position(s) along x
        yy: float or array_like
            Position(s) along y
    Returns:
        R: float or array_like
            Returns the distance of each point (x, y) to (0, 0)
    """
    return abs(np.sqrt(xx**2 + yy**2))


def fft_feature(mmap, smallest_feature, biggest_feature, mmap_step):
    """
    Input:
        mmap: array_like
            This is the libs map data and should be in the for (x_size, y_size, wavelengths)
        smallest_feature: int or float
            Size of the smallest feature to filter
        biggest_feature: int or float
            Size of the biggest feature to filter
        mmap_set: int or float
            Resolution of the LIBS map
    Returns:
        SIR: array_like
            Returns the calculated SIR metric for all wavelengths in the dataset
    """
    freqs_x, freqs_y = get_freqs(mmap, mmap_step)

    fft_map = np.array([np.fft.fftshift(np.fft.fft2(mmap[:, :, i])) for i in range(0, (mmap.shape[-1]))])
    fft_map[:, fft_map.shape[1]//2, fft_map.shape[2]//2] = 0

    kxx, kyy = np.meshgrid(np.fft.fftshift(freqs_x), np.fft.fftshift(freqs_y))

    size_kspace_big = size_relation(biggest_feature, mmap_step)
    size_kspace_small = size_relation(smallest_feature, mmap_step)

    R = dist(kxx, kyy)

    sum1 = np.sum(np.abs(fft_map[:, (size_kspace_big < R) & (R < size_kspace_small)]), axis = (1))
    max1 = np.sum(np.abs(fft_map), axis = (1, 2))

    SIR = np.array(sum1)/np.array(max1)

    return SIR
