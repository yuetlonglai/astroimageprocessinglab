import numpy as np
import scipy as sp
from numpy import fft
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style, ImageNormalize, HistEqStretch, PercentileInterval, AsymmetricPercentileInterval, LinearStretch
from matplotlib import colors


class Process:

    def __init__(self, imgpath) -> None:
        self.img = self.open_img(imgpath)
        pass

    def open_img(self, path):
        hdul = fits.open(path)
        imag = hdul[0].data

        return imag


    def circle_lowpass(self, cutoff, tau):
        self.show_img(self.img)
        circ = self.draw_circle(cutoff, tau)

        imgfft = fft.fft2(self.img)
        imgfft = fft.fftshift(imgfft)

        self.show_img(abs(imgfft))

        print(imgfft)

        
        imgmask = np.multiply(imgfft, circ) 

        self.show_img(circ)
        self.show_img(abs(imgmask))
        

        imgifft = fft.ifft2(imgmask)

        print(imgifft)

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

    def show_img(self, img = None):
        if img is None:
            img = self.img

        fig, ax = plt.subplots(ncols=2)
        ax[0].imshow(img,cmap='inferno', norm=colors.LogNorm())
        ax[1].imshow(img,cmap='Greys',norm=ImageNormalize(img,interval=AsymmetricPercentileInterval(47,53,10),stretch=LinearStretch(5,-4)))

        plt.show()
        




img = Process('A1_mosaic.fits')

lp = img.circle_lowpass(700, 0.001)
img.show_img(lp)

        