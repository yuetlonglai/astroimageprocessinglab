import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd
from astropy.io import fits
from astropy import visualization

# loading the data
hdul = fits.open('A1_mosaic.fits')

# the image
imag = hdul[0].data

# visualising the image
plt.style.use(visualization.astropy_mpl_style)
plt.figure()
plt.imshow(imag,cmap='plasma',norm=colors.LogNorm())
plt.colorbar()
plt.show()

