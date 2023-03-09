import numpy as np
import scipy as sp
from numpy import fft
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy import visualization
from matplotlib import colors
from skimage import segmentation
from skimage import filters
from skimage import measure
from skimage import morphology

class Process:

    def __init__(self, imgpath) -> None:
        # load image
        self.img = self.open_img(imgpath)
        # flip image horizontally to have to 'correct' orientation
        self.img = self.img[::-1]
        # background value from histogram gaussian fit
        self.average_background = 3418.8155053212563 
        self.zpinst, self.zpinst_error = self.flux_calibration(imgpath)
        pass

    def open_img(self, path):
        # for loading image
        hdul = fits.open(path)
        imag = hdul[0].data
        return imag
    
    def flux_calibration(self, path):
        hdul = fits.open(path)
        ZP, ZP_error = hdul[0].header['MAGZPT'], hdul[0].header['MAGZRR']
        return ZP, ZP_error

    def circle_lowpass(self, cutoff, tau):
        self.show_img(self.img)
        circ = self.draw_circle(cutoff, tau)

        imgfft = fft.fft2(self.img)
        imgfft = fft.fftshift(imgfft)

        self.show_img(abs(imgfft))

        # print(imgfft)

        
        imgmask = np.multiply(imgfft, circ) 

        self.show_img(circ)
        self.show_img(abs(imgmask))
        

        imgifft = fft.ifft2(imgmask)

        # print(imgifft)

        return abs(imgifft)

    def find_fit():
        pass

    def draw_circle(self, r, tau):
        canv = np.full(self.img.shape, 0, dtype = float)
        center = np.array(self.img.shape)/2.0
        min_dist = 1e8 
        for i, row in enumerate(canv):
            for j, elem in enumerate(row):
                dist = (i- center[0])**2 + (j - center[1])**2
                if  dist <= r ** 2:
                    canv[i,j] = 1
                else:
                    #val = -tau*np.sqrt(dist - r**2) + 1
                    val = np.exp(-(tau*(np.sqrt(dist - r**2))))
                    if val < 0:
                        val = 0
                    canv[i,j] = val
                    
                    # if min_dist > dist - r**2:
                    #     print(dist - r**2)
                    #     print(val)
                    #     min_dist = ((dist - r**2))

                    #print(min_dist)

        return canv
    
    def clean_bleeding(self, bleedlist, clean_background = True):
        # remove stars that bleeds through the image -> irrelevant to analysis so remove
        # need a list of the coordinates of the stars that bleed in the image
        for i in bleedlist: 
            self.img = segmentation.flood_fill(
                self.img,
                i,
                self.average_background,
                connectivity = 5,
                tolerance = self.img[i[0]][i[1]]-self.average_background-0.5e3
                )
        # clean background? (optional) (pick False if don't wanna clean)
        if clean_background == True:
            self.img = segmentation.flood_fill(self.img,bleedlist[0],self.average_background,connectivity=5,tolerance=1e3)
        return self

    def identify_objects(self):
        # apply threshold
        thresh = 4.5e3 #set threshold of the magnitude of objects
        bw = morphology.closing(self.img > thresh, morphology.disk(6))
        # remove artifacts connected to image border
        cleared = segmentation.clear_border(bw)
        # label image regions
        label_image = measure.label(cleared)
        self.spotsx = []
        self.spotsy = []
        for region in measure.regionprops(label_image):
            # take regions with large enough areas
            if region.area >= 1:
                # draw rectangle around segmented coins
                minr, minc, maxr, maxc = region.bbox
                self.spotsx.append(int((minr+maxr)/2))
                self.spotsy.append(int((minc+maxc)/2))
        return self.spotsx, self.spotsy
    
    def aperture_photometry(self,radius=6):
        y, x = self.identify_objects()
        self.fluxes = []
        for i in range(len(x)): #scanning each detected object's total flux radially in both x and y directions 
            flux = 0
            for k in range(-radius,radius+1): #vertical
                for j in range(-(radius-abs(k)),radius-abs(k)): #horizontal
                    flux += self.img[y[i]+k][x[i]+j]
            self.fluxes.append(flux-self.average_background) #assume constant background contribution for now
        # calibrating the fluxes using ZPinsturment values from header
        self.fluxes = np.array(self.fluxes)
        self.fluxes = self.zpinst - 2.5 * np.log10(self.fluxes) # value
        self.fluxes_error = self.zpinst_error - 2.5 * np.log10(self.fluxes) #value error
        return self.fluxes, self.fluxes_error

    def show_img(self, img = None):
        # plotting the image
        if img is None:
            img = self.img
        normalise = visualization.ImageNormalize(
            img,
            interval=visualization.AsymmetricPercentileInterval(47,53,6),
            stretch=visualization.LinearStretch(5,-4)
            )
        fig, ax = plt.subplots(ncols=2,figsize=(10,8))
        ax[0].imshow(img,cmap='inferno', norm=colors.LogNorm())
        ax[1].imshow(img,cmap='Greys',norm=normalise)
        plt.show()
        




image = Process('A1_mosaic.fits')

# lp = img.circle_lowpass(700, 0.001)
bleeding = [(1400,1438),(514,558),(578,1456),(851,2133),(1292,776),(1196,2465),(1838,973),(2325,905),(2301,2131),(3184,2088),(4602,1431),(4488,1531)]
image.clean_bleeding(bleeding,clean_background=False)
# image.show_img()
y,x = image.identify_objects()
fluxlist = image.aperture_photometry()
print(fluxlist)
plt.figure(figsize=(10,8))
plt.imshow(image.img,cmap='plasma',norm=colors.LogNorm())
plt.plot(x,y,'x',color='yellow',alpha=0.2)
plt.show()
        