import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colors
from scipy import signal
from scipy import ndimage
from scipy import fftpack
from astropy import io
from astropy import visualization
from skimage import segmentation
from skimage import filters
from skimage import measure
from skimage import morphology

# loading the data
hdul = io.fits.open('A1_mosaic.fits')

# the image
imag = hdul[0].data
# print(imag)
imag = imag[::-1]
imag1 = np.reshape(imag,len(imag)*len(imag[0]))

# background
average_background = 3418.8155053212563 # from histogram gaussian fit

# slicing image
# imaghslice = imag[2000]
# imaghslice_lowpass = signal.savgol_filter(imaghslice,window_length=51,polyorder=3)
# imaghslice_peaks, props = signal.find_peaks(imaghslice_lowpass,height=3460,distance=10)
# plt.figure()
# plt.xlabel('Pixel')
# plt.ylabel('Magnitude')
# plt.plot(range(len(imaghslice)),imaghslice,'-',color='royalblue')
# # plt.plot(range(len(imaghslice)),imaghslice_lowpass,'-',color='cyan')
# plt.hlines(y=average_background,xmax=max(range(len(imaghslice))),xmin=min(range(len(imaghslice))),color='black',linestyles='dashed')
# # plt.plot(np.array(range(len(imaghslice)))[imaghslice_peaks],imaghslice_lowpass[imaghslice_peaks],'.',color='red')
# plt.ticklabel_format(style='sci',scilimits=(0,0))
# plt.show()

# bleeding = [(1438,1400),(558,514),(1456,578),(2133,851),(776,1292),(2465,1196),(973,1838),(905,2325),(2131,2301),(2088,3184)]
bleeding = [(1400,1438),(514,558),(578,1456),(851,2133),(1292,776),(1196,2465),(1838,973),(2325,905),(2301,2131),(3184,2088),(4602,1431),(4488,1531)]

plt.figure(figsize=(10,8))
plt.subplot(1,2,1)
plt.title('Before')
plt.imshow(imag,cmap='plasma', norm = colors.LogNorm())
plt.subplot(1,2,2)
plt.title('After')
for i in bleeding:
    imag = segmentation.flood_fill(imag,i,average_background,connectivity=5,tolerance=imag[i[0]][i[1]]-average_background-1e3)
imag = segmentation.flood_fill(imag,bleeding[0],average_background,connectivity=5,tolerance=1e3)
# apply threshold
thresh = filters.threshold_otsu(imag)
bw = morphology.closing(imag > thresh, morphology.square(3))
# remove artifacts connected to image border
cleared = segmentation.clear_border(bw)
# label image regions
label_image = measure.label(cleared)
spotsx = []
spotsy = []
for region in measure.regionprops(label_image):
    # take regions with large enough areas
    if region.area >= 1:
        # draw rectangle around segmented coins
        minr, minc, maxr, maxc = region.bbox
        spotsx.append((minr+maxr)/2)
        spotsy.append((minc+maxc)/2)
        # rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
        # ax.add_patch(rect)

plt.imshow(imag,cmap='plasma', norm = colors.LogNorm())
plt.plot(spotsy,spotsx,'x',color='yellow',alpha=0.3)
plt.show()

# plt.figure(figsize=(12,8))
# plt.subplot(1,2,1)
# plt.imshow(imag,cmap='plasma',norm=colors.LogNorm())
# plt.subplot(1,2,2)
# plt.imshow(imag,cmap='Greys',norm=visualization.ImageNormalize(imag,interval=visualization.AsymmetricPercentileInterval(49,53,10),stretch=visualization.LinearStretch(5,-5)))
# plt.colorbar(label='Pixel Values')
# plt.show()


