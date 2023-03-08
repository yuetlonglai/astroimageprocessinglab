import numpy as np
import scipy as sp
from numpy import fft
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy import visualization
from matplotlib import colors
from skimage import segmentation


class Process:

    def __init__(self, imgpath) -> None:
        # load image
        self.img = self.open_img(imgpath)
        # flip image horizontally to have to 'correct' orientation
        self.img = self.img[::-1]
        # background value from histogram gaussian fit
        self.average_background = 3418.8155053212563 
        pass

    def open_img(self, path):
        # for loading image
        hdul = fits.open(path)
        imag = hdul[0].data
        return imag

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
    
    def clean_bleeding(self, bleedlist):
        # remove stars that bleeds through the image -> irrelevant to analysis so remove
        # need a list of the coordinates of the stars that bleed in the image
        for i in bleedlist: 
            self.img = segmentation.flood_fill(
                self.img,
                i,
                self.average_background,
                connectivity = 5,
                tolerance = self.img[i[0]][i[1]]-self.average_background-1e3
                )

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
bleeding = [(1400,1438),(514,558),(578,1456),(851,2133),(1292,776),(1196,2465),(1838,973),(2325,905),(2301,2131),(3184,2088),(4602,1431)]
image.clean_bleeding(bleeding)
image.show_img()

        