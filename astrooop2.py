import numpy as np
import pandas as pd
from numpy import fft
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy import visualization
from astropy import stats
from matplotlib import colors
from skimage import segmentation
from photutils import detection
from photutils import background
from photutils import aperture

class Process2:
    def __init__(self, imgpath) -> None:
        # load image
        self.img = self.open_img(imgpath)
        # flip image horizontally to have to 'correct' orientation
        self.img = self.img[::-1]
        self.original = self.img
        # background value from histogram gaussian fit
        self.average_background = 3418.8155053212563 
        self.background_sigma = 17.477502359953714
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
    
    def mask_edge(self, cut):
        cutimg = self.img[cut:-cut, cut:-cut]
        img2 = np.pad(cutimg, ((cut,cut),(cut,cut)), mode='constant', constant_values = (self.average_background))
        # self.show_img()
        return img2
    
    def clean_bleeding(self, bleedlist, clean_background = False):
        # remove stars that bleeds through the image -> irrelevant to analysis so remove
        # need a list of the coordinates of the stars that bleed in the image
        for i in bleedlist: 
            self.img = segmentation.flood_fill(
                self.img,
                i,
                self.average_background,
                connectivity = 5,
                tolerance = self.img[i[0]][i[1]]-self.average_background-0.05e3
                )
        # clean background? (optional) (pick False if don't wanna clean)
        if clean_background == True:
            self.img = segmentation.flood_fill(self.img,bleedlist[0],self.average_background,connectivity=5,tolerance=1e3)
            # self.img = np.where(self.img < 100, 0, self.img)
        return self
    
    def identify_objects(self,catalogue=False):
        # remove background
        self.bkg = background.Background2D(self.img,(50,50),filter_size=(3,3),sigma_clip=stats.SigmaClip(sigma=3.0),bkg_estimator=background.MedianBackground())
        self.img = self.img - self.bkg.background
        # finding sources
        daofind = detection.DAOStarFinder(fwhm=7.0,threshold=5*self.background_sigma)
        sources = daofind(self.mask_edge(150))
        print('Number of Object Detected : ' + str(len(sources)))
        sourcesx = sources['xcentroid']
        sourcesy = sources['ycentroid']
        # aperture photometry
        positions = np.column_stack([sourcesx,sourcesy])
        apertures = [aperture.CircularAperture(positions, r=r) for r in [6.0,8.0,10.0]]
        phot_table = aperture.aperture_photometry(self.img, apertures)
        if catalogue == False:
            return sourcesy, sourcesx
        else:
            catalogue_table = pd.DataFrame(columns=['x-centroid','y-centroid'])
            catalogue_table['x-centroid'] = sourcesx
            catalogue_table['y-centroid'] = sourcesy
            catalogue_table['magnitude_0'] = self.zpinst - 2.5*np.log10(phot_table['aperture_sum_0'])
            catalogue_table['magnitude_1'] = self.zpinst - 2.5*np.log10(phot_table['aperture_sum_1'])
            catalogue_table['magnitdue_2'] = self.zpinst - 2.5*np.log10(phot_table['aperture_sum_2'])
            return catalogue_table
        
    def number_count(self, plotting=False):
        cat = self.identify_objects(catalogue=True)
        magnitude_radius = cat['magnitude_0']
        m = np.linspace(min(magnitude_radius),18,50)
        N = []
        Nerr= []
        for i in m:
            N.append(np.count_nonzero(np.where(magnitude_radius < i, True, False)))
            # Nerr_l.append(np.count_nonzero(np.where(cat['magnitude']-abs(cat['magnitude-error']) < i, True, False)))
            # Nerr_u.append(np.count_nonzero(np.where(cat['magnitude']+abs(cat['magnitude-error']) < i, True, False)))
        if plotting == True:
            print(N)
            fit,cov = np.polyfit(m[1:],np.log10(N[1:]),1,cov=True)
            logN = np.poly1d(fit)
            print('Gradient = %.3e +/- %.3e' %(fit[0],np.sqrt(cov[0][0])))
            plt.figure()
            plt.xlabel(r'$m$')
            plt.ylabel(r'$log(N(m))$')
            # plt.errorbar(x=counting[0],y=np.log10(counting[1]),yerr=(np.log10(counting[2]),np.log10(counting[3])),fmt='.',capsize=2,color='blue',label='Data')
            plt.errorbar(x=m[1:],y=np.log10(N[1:]),fmt='.',color='blue',label='Data')
            plt.plot(m[1:],logN(m[1:]),'-',color='red',label='Fitted')
            plt.legend(loc='best')
            plt.show()
        return m, N#, abs(np.array(N)-np.array(Nerr_l)), abs(np.array(Nerr_u)-np.array(N))
        

# using the class
image = Process2('A1_mosaic.fits')
# image.double_lowpass()
bleeding = [(1400,1438),(514,558),(578,1456),(851,2133),(1292,776),(1196,2465),(1838,973),(2325,905),(2301,2131),(3184,2088),(4602,1431),(4488,1531)]
image.clean_bleeding(bleeding,clean_background=False)
y,x = image.identify_objects()
plt.figure(figsize=(10,8))
normalise = visualization.ImageNormalize(image.img,interval=visualization.AsymmetricPercentileInterval(5,95),stretch=visualization.LinearStretch())
plt.imshow(image.img,cmap='Greys',norm=normalise)
# plt.imshow(image.img,cmap='inferno',norm=colors.LogNorm())
plt.plot(x,y,'.',color='red',alpha=0.5)
plt.colorbar()
plt.show()

image.number_count(plotting=True)