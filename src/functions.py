import numpy as np

class Tools:
    def get_freqs(mmap, mmap_step):
        nx, ny = mmap.shape[1], mmap.shape[0]
        dx, dy = mmap_step, mmap_step

        freqs_x = 2*np.pi*np.fft.fftfreq(nx, dx)
        freqs_y = 2*np.pi*np.fft.fftfreq(ny, dy)
        
        return freqs_x, freqs_y


    def size_relation(size, mmap_step):
        object_size = size*mmap_step
        size_kspace = 2*np.pi/object_size/2

        return size_kspace

    def radial_dist(x, y):
        return abs(np.sqrt(x**2 + y**2))


class FFT_metric:
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
                Returns the SIR calculated SIR metric for all wavelengths in the dataset
        """
        freqs_x, freqs_y = Tools.get_freqs(mmap, mmap_step)

        fft_map = np.array([np.fft.fftshift(np.fft.fft2(mmap[:, :, i])) for i in range(0, (mmap.shape[-1]))])
        fft_map[:, fft_map.shape[1]//2, fft_map.shape[2]//2] = 0

        kxx, kyy = np.meshgrid(np.fft.fftshift(freqs_x), np.fft.fftshift(freqs_y))

        size_kspace_big = Tools.size_relation(biggest_feature, mmap_step)
        size_kspace_small = Tools.size_relation(smallest_feature, mmap_step)

        R = Tools.radial_dist(kxx, kyy)

        sum1 = np.sum(np.abs(fft_map[:, (size_kspace_big < R) & (R < size_kspace_small)]), axis = (1))
        max1 = np.sum(np.abs(fft_map), axis = (1, 2))

        SIR = np.array(sum1)/np.array(max1)

        return SIR
