import numpy as np
import h5py as h5

def read_data_map(file):
  """
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
  # fft_scores = np.array(hf['System properties']['fft_score'])
  
  
  unique_x = np.unique(positions[:,0])
  unique_y = np.unique(positions[:,1])
  
  nx = len(unique_x)
  ny = len(unique_y)
  indexes = np.lexsort((positions[:, 0],positions[:, 1]))
  spectrums = spectrums[indexes, :]
  spectrums = np.array(spectrums.reshape((ny, nx, -1)))
  # print(spectrums.shape)
  positions = np.array(positions[indexes])
  
  hf.close()
  return spectrums, wavelengths, positions, unique_x, unique_y#, fft_scores
