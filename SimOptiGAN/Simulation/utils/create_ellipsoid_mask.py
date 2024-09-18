import numpy as np
from skimage.draw import ellipsoid
import tifffile as tiff

'''Script to create an ellipsoid mask image which can be used during simulation.'''

output_path = './data/masks/ellipsoid_mask.tif'
a, b, c = 200, 200, 200


ellip = ellipsoid(a, b, c, spacing=(1.0, 1.0, 1.0))
ellip_small = ellipsoid(a*2/3, b*2/3, c*2/3, spacing=(1.0, 1.0, 1.0))
pad = np.round(np.subtract(ellip.shape,ellip_small.shape)/2).astype(int)
pad = np.stack([pad, pad], axis=1)
print(pad)
ellip_small = np.pad(ellip_small, pad)

ellip = ellip ^ ellip_small

tiff.imwrite(output_path, ellip)