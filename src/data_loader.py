import numpy as np
import h5py as h5

def read_data_map(file):
  """
  Input:
        file: string
            Location of the LIBS map file
    Returns:
        spectrum: array_like (x_size, y_size, wavelengths)
        wavelengths: array_like (num_wavelengths)
        positions: array_like (2, num_shots)
        unique_x: array_like (x_size)
        unique_y: array_like (y_size)
            Returns the LIBS data cube, along with the spectral wavelengths, the shot positions and the unique positions along each map axis
  """

  hf = h5py.File(file + '.h5', 'r')
  sample = file.split("\\")[-1]
  keys = [key for key in hf.keys()]
  sample = keys[0].split(' ')[-1]
  
  spectrums = np.array([np.ndarray.flatten(np.array(list(hf['Sample_ID: ' + sample]['Spot_' + str(i)]['Shot_0']['Pro']))) for i in range(0,
                                                      len(list(hf['Sample_ID: ' + sample])))])
  positions = np.array([np.ndarray.flatten(np.array(list(hf['Sample_ID: ' + sample]['Spot_' + str(i)]['position']))) for i in range(0,
                                                      len(list(hf['Sample_ID: ' + sample])))])
  wavelengths = np.array(hf['System properties']['wavelengths'])
  
  
  unique_x = np.unique(positions[:,0])
  unique_y = np.unique(positions[:,1])
  
  nx = len(unique_x)
  ny = len(unique_y)
  indexes = np.lexsort((positions[:, 0],positions[:, 1]))
  spectrums = spectrums[indexes, :]
  spectrums = np.array(spectrums.reshape((ny, nx, -1)))
  positions = np.array(positions[indexes])
  
  hf.close()
  return spectrums, wavelengths, positions, unique_x, unique_y
