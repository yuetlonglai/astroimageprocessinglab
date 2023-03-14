import numpy as np
import pandas as pd
from numpy import fft
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy import visualization
from matplotlib import colors
from skimage import segmentation
from skimage import measure
from skimage import morphology
from skimage import restoration
from skimage import filters

# loading the data
hdul = fits.open('A1_mosaic.fits')

# the image
imag = hdul[0].data
# print(imag)
imag = imag[::-1]

# background
average_background = 3418.8155053212563 # from histogram gaussian fit

bleeding = [(1400,1438),(514,558),(578,1456),(851,2133),(1292,776),(1196,2465),(1838,973),(2325,905),(2301,2131),(3184,2088),(4602,1431),(4488,1531)]
for i in bleeding:
    imag = segmentation.flood_fill(imag,i,average_background,connectivity=5,tolerance=imag[i[0]][i[1]]-average_background-0.5e3)

cut = 200
cutimg = imag[cut:-cut, cut:-cut]
imag2 = np.pad(cutimg, ((cut,cut),(cut,cut)), mode='constant', constant_values = (average_background))
# stars = morphology.local_maxima(imag2,connectivity=10,indices=True,allow_borders=False)

# print(stars)
# peaksx = []
# peaksy = []
# for j in range(len(stars)):
#     for i in range(len(stars[0])):
#         if stars[j][i] == 1:
#             peaksx.append(i)
#             peaksy.append(j)


plt.figure(figsize=(10,8))
# plt.imshow(imag,cmap='inferno',norm=colors.LogNorm())
normalise = visualization.ImageNormalize(imag,interval=visualization.AsymmetricPercentileInterval(49,53,11),stretch=visualization.LinearStretch(2,-1))
plt.imshow(imag,cmap='Greys', norm = normalise)
# plt.plot(stars[1],stars[0],'x',color='yellow',alpha=0.5)
plt.colorbar()
plt.show()

hdul.close()