import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize 
import pandas as pd
from astropy.io import fits
from astropy.visualization import astropy_mpl_style, ImageNormalize, HistEqStretch, PercentileInterval, LinearStretch

# loading the data
hdul = fits.open('A1_mosaic.fits')

# the image
imag = hdul[0].data
# print(imag)
imag = imag[::-1]

# visualising the statistics of the image
imag1 = np.reshape(imag,len(imag)*len(imag[0]))
# print(min(imag1),max(imag1))
plt.figure()
plt.xlabel('Pixel Values')
plt.ylabel('Count')
bin_heights, bin_edges, patches = plt.hist(imag1,bins=20000)
midpoints = 0.5 * (bin_edges[1:]+bin_edges[:-1])
def gaussian(x,a,b,c,d):
    return a*np.exp(-(x-b)**2/c**2) +d
popt,pcov = optimize.curve_fit(gaussian,midpoints,bin_heights,p0=[1e6,3421,10,0])
plt.plot(midpoints,bin_heights,'x',color='cyan')
pix = np.linspace(3.3e3,3.6e3,1000)
plt.plot(pix,gaussian(pix,*popt),'-',color='salmon')
plt.xlim(3.3e3,3.6e3)
plt.show()

# visualising the image
# plt.style.use(astropy_mpl_style)
plt.figure(figsize=(5,8))
plt.imshow(imag,cmap='Greys',norm=ImageNormalize(imag,interval=PercentileInterval(0.1,16),stretch=LinearStretch(2,-1)))
# plt.imshow(imag,cmap='Greys',norm=ImageNormalize(imag,interval=PercentileInterval(0.1,10),stretch=HistEqStretch(imag)))
cbar = plt.colorbar()
plt.show()

