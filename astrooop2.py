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
from scipy import optimize
from scipy.interpolate import RegularGridInterpolator
import pickle
from tensorflow.keras import models, Sequential
import tensorflow as tf
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
    
    def clean_bleeding(self, bleedlist):
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
        return self
    
    def identify_objects(self,catalogue=False,aperture_vary=True,sersic=False): # use photoutils aperture, photometry, background, detection
        # remove background
        self.bkg = background.Background2D(self.img,(50,50),filter_size=(3,3),sigma_clip=stats.SigmaClip(sigma=3.0),bkg_estimator=background.MedianBackground())
        self.img = self.img - self.bkg.background
        # mask
        mask = np.ones(self.img.shape, dtype=bool)
        mask[150:-150, 150:-150] = False
        # finding sources
        daofind = detection.DAOStarFinder(fwhm=7.0,threshold=5*self.background_sigma)
        sources = daofind(self.img, mask=mask)
        sourcesx = sources['xcentroid']
        sourcesy = sources['ycentroid']
        sources_radius = 5.0 * np.log10(sources['peak']) - 5.0 # approximate the relationship between brightness and its radius => works
        positions = np.column_stack([sourcesx,sourcesy])
        ellipticity = sources['roundness1']
        # aperture photometry
        if aperture_vary == False:
            # with fixed aperture(s)
            multi_apertures = [aperture.CircularAperture(positions, r=r) for r in [6.0,8.0,10.0]]
            self.apertures = aperture.CircularAperture(positions, r=6.0)
            phot_table = aperture.aperture_photometry(self.img, multi_apertures, method='subpixel')
        else:
            # with varying apertures
            varying_phot = []
            self.apertures = []
            for i in range(len(positions)):
                # individual_apertures = aperture.CircularAperture(positions[i],r=sources_radius[i])
                individual_apertures = aperture.EllipticalAperture(positions[i],a=sources_radius[i]*(1+abs(ellipticity[i])/2),b=sources_radius[i]*(1-abs(ellipticity[i])/2),theta=np.arctan(ellipticity[i]))
                photo = aperture.aperture_photometry(self.img,individual_apertures, method='subpixel')
                varying_phot.append(photo['aperture_sum'])
                self.apertures.append(individual_apertures)
        # sersic profile and return
        # if sersic == True:
        #     nvalues = []
        #     galaxies_x = []
        #     galaxies_y = []
        #     varying_phot_sersic = []
        #     for i in range(len(positions)):
        #         radial = np.linspace(0.001,sources_radius[i],20)
        #         annulal_flux = []
        #         radial_mid = []
        #         for j in range(len(radial)-1):
        #             individual_annulus = aperture.EllipticalAnnulus(
        #                 positions[i],
        #                 a_in = radial[j]*(1+abs(ellipticity[i])/2),
        #                 a_out = radial[j+1]*(1+abs(ellipticity[i])/2),
        #                 b_in = radial[j]*(1-abs(ellipticity[i])/2),
        #                 b_out = radial[j+1]*(1-abs(ellipticity[i])/2)
        #             )
        #             flux = aperture.aperture_photometry(self.img,individual_annulus,method='subpixel')
        #             annulal_flux.append(flux['aperture_sum'][0])
        #             radial_mid.append((radial[j]+radial[j+1])/2)
        #         try:
        #             popt,pcov = optimize.curve_fit(self.log_sersic,radial_mid,annulal_flux,p0=[abs(sources['peak'][i]),1e3,1.0])
        #         except:
        #             pass
        #         if popt[-1] > 0.5 and popt[-1] < 10:
        #             nvalues.append(popt[-1])
        #             galaxies_x.append(positions[i][0])
        #             galaxies_y.append(positions[i][1])
        #             individual_apertures = aperture.EllipticalAperture(positions[i],a=sources_radius[i]*(1+abs(ellipticity[i])/2),b=sources_radius[i]*(1-abs(ellipticity[i])/2),theta=np.arctan(ellipticity[i]))
        #             photo = aperture.aperture_photometry(self.img,individual_apertures, method='subpixel')
        #             varying_phot_sersic.append(photo['aperture_sum'])
        #     if catalogue == False:
        #         return galaxies_y, galaxies_x
        #     else:
        #         catalogue_table = pd.DataFrame(columns=['x-centroid','y-centroid'])
        #         catalogue_table['x-centroid'] = galaxies_x
        #         catalogue_table['y-centroid'] = galaxies_y
        #         catalogue_table['magnitude_'] = self.zpinst - 2.5*np.log10(np.array(varying_phot_sersic))
        #         return catalogue_table
        # return method
        if catalogue == False:
            print('Number of Object Detected : ' + str(len(sources)))
            return sourcesy, sourcesx
        else:
            catalogue_table = pd.DataFrame(columns=['x-centroid','y-centroid'])
            catalogue_table['x-centroid'] = sourcesx
            catalogue_table['y-centroid'] = sourcesy
            catalogue_table['peaks'] = sources['peak']
            if aperture_vary == False:
                catalogue_table['magnitude_0'] = self.zpinst - 2.5*np.log10(phot_table['aperture_sum_0'])
                catalogue_table['magnitude_1'] = self.zpinst - 2.5*np.log10(phot_table['aperture_sum_1'])
                catalogue_table['magnitude_2'] = self.zpinst - 2.5*np.log10(phot_table['aperture_sum_2'])
            else:
                catalogue_table['magnitude_'] = self.zpinst - 2.5*np.log10(np.array(varying_phot))
            return catalogue_table
        
    def number_count(self, plotting=False,num='',ser=False, cat = None):
        if cat is None:
            if ser == False:
                cat = self.identify_objects(catalogue=True)
            else:
                cat = self.identify_objects(catalogue=True,sersic=True)
        aperture_num = num
        magnitude_radius = cat['magnitude_'+str(aperture_num)]
        m = np.linspace(min(magnitude_radius),max(magnitude_radius),50)
        N = []
        Nerr= []
        for i in m:
            N.append(np.count_nonzero(np.where(magnitude_radius < i, True, False)))
            # Nerr_l.append(np.count_nonzero(np.where(cat['magnitude']-abs(cat['magnitude-error']) < i, True, False)))
            # Nerr_u.append(np.count_nonzero(np.where(cat['magnitude']+abs(cat['magnitude-error']) < i, True, False)))
        N = np.array(N)
        m, N = m[1:], N[1:]
        logN = np.log10(N)
        logNerr = (1/N * N**0.5)*logN
        if plotting == True:
            print(N)
            fit,cov = np.polyfit(m[:-15],logN[:-15],1,w=1/logNerr[:-15],cov=True)
            plogN = np.poly1d(fit)
            print('Gradient = %.3e +/- %.3e' %(fit[0],np.sqrt(cov[0][0])))
            plt.figure()
            plt.xlabel(r'$m$')
            plt.ylabel(r'$log(N(m))$')
            plt.errorbar(x=m,y=logN,fmt='.',yerr=logNerr,capsize=2,label='Data '+str(aperture_num),color='blue')
            plt.plot(m,plogN(m),'-',color='red',label='Fitted')
            plt.legend(loc='best')
            plt.show()
        return m, N#, abs(np.array(N)-np.array(Nerr_l)), abs(np.array(Nerr_u)-np.array(N))
    
    def log_sersic(self,r,I0,k,n):
        return np.log(I0) - k*r**(1/n)
    
    def objwindow(self, x, y, coords, rng, wscale, wdim):
        xmask = (coords[0] - wscale* rng < x) & (x < coords[0] +  wscale* rng)
        ymask = (coords[1] - wscale * rng < y) & (y < coords[1] + wscale * rng)


        window = self.img[ymask][:,xmask]
        
        winterp = RegularGridInterpolator((y[ymask],x[xmask]),window)


        nx = np.linspace(min(x[xmask]),max(x[xmask]),wdim[0])
        ny = np.linspace(min(y[ymask]),max(y[ymask]),wdim[1])

        NX, NY = np.meshgrid(ny,nx)
        newdat = winterp((NX,NY))
        newdat = (newdat - np.amin(newdat))/np.amax(newdat - np.amin(newdat))
        newdat = np.reshape(newdat,(64,64,1))
        return newdat
    
    def gal_ml(self, model, cat):
        x = np.linspace(0,self.img.shape[1],self.img.shape[1])
        y = np.linspace(0,self.img.shape[0],self.img.shape[0])
        coordlist = list(zip(cat['x-centroid'],cat['y-centroid']))
        
        winds = []
        for i, (coord, peak) in enumerate(zip(coordlist, cat['peaks'])):
            width = 5 * np.log(peak) - 5
            print(width)
            wind = self.objwindow(x,y,coord, width, 2, [64,64])
            winds.append(wind)
        
        winds = np.array(winds)

        predicts = model.predict_on_batch(winds)

        return (winds, predicts)
    
    def show_img(self, img = None):
        # plotting the image
        if img is None:
            img = self.img
        normalise = visualization.ImageNormalize(
            img,
            interval=visualization.AsymmetricPercentileInterval(5,95),
            stretch=visualization.LinearStretch()
            )
        fig, ax = plt.subplots(ncols=2,figsize=(10,8))
        ax[0].imshow(img,cmap='inferno', norm=colors.LogNorm())
        ax[1].imshow(img,cmap='Greys',norm=normalise)
        plt.show()


    
if __name__ == '__main__':
    # using the class
    image = Process2('A1_mosaic.fits')
    # clean bleeding
    bleeding = [(1400,1438),(514,558),(578,1456),(851,2133),(1292,776),(1196,2465),(1838,973),(2325,905),(2301,2131),(3184,2088),(4602,1431),(4488,1531),(4183,1036),(4275,1642)]
    image.clean_bleeding(bleeding)
    # identify objects
    y,x = image.identify_objects()
    # plot
    plt.figure(figsize=(10,8))
    normalise = visualization.ImageNormalize(image.img,interval=visualization.AsymmetricPercentileInterval(5,95),stretch=visualization.LinearStretch())
    plt.imshow(image.img,cmap='Greys',norm=normalise)
    # plt.imshow(image.original,cmap='inferno',norm=colors.LogNorm())
    plt.plot(x,y,'.',color='red',alpha=0.5)
    for i in range(len(image.apertures)):
        image.apertures[i].plot(color='blue', lw=1.5, alpha=0.5)
    plt.colorbar()
    plt.show()

    # cumulative number count plot


    model = models.load_model('TrainedModel.h5')

    probability_model = Sequential([model, tf.keras.layers.Softmax()])
    
    cat = image.identify_objects(catalogue = True)

    winds, predicts = image.gal_ml(probability_model,cat)

    cat['galpreds'] = predicts[:,1]

    galaxy_dat = cat.query('galpreds > 0.95')
    plt.figure(figsize=(10,8))

    normalise = visualization.ImageNormalize(image.img,interval=visualization.AsymmetricPercentileInterval(5,95),stretch=visualization.LinearStretch())
    plt.imshow(image.img,cmap='Greys',norm=normalise)
    plt.scatter(cat['x-centroid'],cat['y-centroid'], s=2, color = 'r')
    plt.scatter(galaxy_dat['x-centroid'],galaxy_dat['y-centroid'], color = 'b', s=2)
    plt.show()

    
    
    m,N=image.number_count(plotting=False)
    mg, Ng = image.number_count(plotting = True,cat = galaxy_dat)

    # m1,N1=image.number_count(plotting=False,ser=True)
    plt.figure()
    plt.scatter(m,np.log(N),color='blue',label='All objects')
    plt.scatter(mg,np.log(Ng),color='blueviolet',label='Galaxies')
    plt.legend()
    plt.show()



    


    
    
    # for i, (wind, predict) in enumerate(zip(winds, predicts)):
    #     image.show_img(wind)
    #     print(predict)

    # print('Galaxy = ' +str(N1[-1]))
    # print('Not Galaxy = ' +str(N[-1]-N1[-1]))