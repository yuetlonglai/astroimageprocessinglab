import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy import optimize 
import pandas as pd
from astropy.io import fits
from astropy.visualization import astropy_mpl_style, ImageNormalize, HistEqStretch, PercentileInterval, AsymmetricPercentileInterval, LinearStretch

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
plt.plot(pix,gaussian(pix,*popt),'-',color='salmon', label = r'$\mu = 3418, \sigma = 17$')
# plt.xlim(3.3e3,3.6e3)
plt.show()
average_background, average_background_err = popt[1], np.sqrt(pcov[1][1])
background_sigma = popt[2]
print(average_background, average_background_err, background_sigma)

# visualising the image in 3d
# average_background = 3418.8155053212563 # from histogram gaussian fit
# imag2 = imag
# for i in range(len(imag2)):
#     for j in range(len(imag2[0])):
#         if j > 1e4:
#             j = average_background
# fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
# X, Y = np.meshgrid(np.array(range(len(imag2[0]))),np.array(range(len(imag2))))
# ax.plot_surface(X,Y,imag2,cmap='plasma')
# # fig.colorbar()
# plt.show()

# visualising the image 
# plt.style.use(astropy_mpl_style)
# plt.figure(figsize=(12,8))
# plt.subplot(1,2,1)
# plt.imshow(imag,cmap='plasma',norm=colors.LogNorm())
# plt.subplot(1,2,2)
# plt.imshow(imag,cmap='Greys',norm=ImageNormalize(imag,interval=AsymmetricPercentileInterval(49,53,8),stretch=LinearStretch(2,-1)))
# plt.colorbar(label='Pixel Values')
# plt.show()

