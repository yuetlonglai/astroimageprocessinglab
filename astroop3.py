from astrooop2 import Process2
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
from perlin_noise import PerlinNoise
#from lmfit.models import Gaussian2dModel
from tqdm import tqdm
import PIL
from glob import glob
import pickle
from scipy.interpolate import RegularGridInterpolator
import matplotlib.cm as cm
import matplotlib.gridspec as gs

class Process_fake(Process2):
    def __init__(
        cls, size = [100,100], noise_params = {'octaves': 5, 'seed': 1, 'scale': 10, 'shift': 10, 'whiterange': 36, 'whiteshift': 5}, 
        objparams = {'numgal': 10, 'numstar': 10, 'galamp': 4000, 'staramp': 4000, 'galwidth': 20, 'gnrange': [3,7], 'starwidth': 10}) -> None:


        noise = PerlinNoise(octaves = noise_params['octaves'], seed = noise_params['seed'])
        noisegrid = np.zeros((100,100))
        for i in tqdm(range(100)):
            for j in range(100):
                noisegrid[i,j] = (noise([i/100, j/100]))*noise_params['scale'] + noise_params['shift']

        #noisegrid = np.random.rand(noise_params['octaves'],noise_params['octaves'])*noise_params['scale'] + noise_params['shift']
        nx = np.linspace(0,100, 100)
        ny = np.linspace(0,100, 100)
        ninter = RegularGridInterpolator((nx,ny),noisegrid)
        x = np.linspace(0,100,size[0])
        y = np.linspace(0,100,size[1])

        X, Y = np.meshgrid(x, y, indexing='ij')



        pic = ninter((X,Y))
        cls.noise1 = pic
        wnoise = np.random.normal(noise_params['whiteshift'],noise_params['whiterange'],(size[0],size[1]))
        pic1 = pic + wnoise
        cls.noise2 = pic1

        cls.img = pic1.copy()
        print(cls.img)
        print(cls.img.shape)
        cls.show_img()
        cls.galprops =  cls.rand_galaxy(objparams['numgal'],objparams['galamp'],objparams['galwidth'], objparams['galskew'], objparams['gnrange'])
        cls.starprops = cls.rand_star(objparams['numstar'],objparams['staramp'],objparams['starwidth'])
        cls.average_background = noise_params['shift']
        cls.background_sigma = noise_params['scale']/2
        cls.zpinst = 25.3
        cls.zpinst_error = 0.02
        cls.show_img()


    def rand_galaxy(self, num, ampr, widthr, skew, nrange):
        xpoints = np.random.randint(len(self.img[0]),size = num)
        ypoints = np.random.randint(len(self.img[:,0]),size = num)

        xrs = np.random.randint(min(widthr),max(widthr),size = num)
        
        skews = abs(np.random.rand(num))*abs(skew[0]-skew[1]) + min(skew)
        amps = np.random.randint(min(ampr),max(ampr), size = num)
        #xrs = 5*np.log(amps) -5

        x = np.arange(0,len(self.img[0]),1, dtype=int)
        y = np.arange(0,len(self.img[1]),1, dtype=int)

        ns = np.random.rand(num) * (max(nrange) - min(nrange)) + min(nrange)

        
        coords = []
        swidths = []
        flux = []
        for i, (x0,y0,xr,s,n,a) in tqdm(enumerate(zip(xpoints,ypoints,xrs,skews,ns,amps))):
            
            angle = np.random.randint(0,360,1)

            xmask = (x < x0 + 10*xr) & (x > x0 - 10*xr)
            ymask = (y0 - 10*xr < y) & (y < y0 + 10*xr)

            xm = x[xmask]
            ym = y[ymask]

            if xm.size == ym.size and xm.size > 1:

                xx, yy = np.meshgrid(xm,ym)
                sg = self.sersic(xx,yy,x0,y0,a,xr,n,s)
                fl = sum(sg.flatten())
                sers = np.pad(sg, ([min(xm),x.size - min(xm) + x.size],[min(ym),y.size - min(ym)+ym.size] ))
                sers = sers[:x.size,:y.size]

                gimg = PIL.Image.fromarray(sers)
                gimg = gimg.rotate(angle, center = [y0,x0])
                #print(x0,y0,xr,s,n,a)
                # plt.imshow(sers, norm=colors.LogNorm())
                # plt.scatter(y0,x0, color = 'r',alpha=0.5)
                # plt.show()

                coords.append(np.array([x0,y0]))
                swidths.append(np.array([xr,xr/s]))
                flux.append(fl)
                self.img += np.array(gimg)

        return {'coords': np.array(coords), 'widths': np.array(swidths), 'amps': amps, 'flux': flux}

    def rand_star(self, num, maxamp, maxwidth):
        xpoints = np.random.randint(self.img.shape[0],size = num)
        ypoints = np.random.randint(self.img.shape[0],size = num)
        
        widths = np.random.randint(min(maxwidth),max(maxwidth),size = num)#

        amps = np.random.randint(min(maxamp),max(maxamp), size = num)
        #widths = np.log(amps) + 5

        x = np.arange(0,len(self.img[0]),1, dtype=int)
        y = np.arange(0,len(self.img[1]),1, dtype=int)

        coords = []
        swidths = []
        flux = []
        for i, (x0,y0,sig,a) in tqdm(enumerate(zip(xpoints,ypoints,widths,amps))):

            xmask = (x < x0 + 5*sig) & (x > x0 - 5*sig)
            ymask = (y0 - 5*sig < y) & (y < y0 + 5*sig)

            xm = x[xmask]
            ym = y[ymask]

            if ym.size == xm.size and xm.size > 1: 

                xx, yy = np.meshgrid(xm,ym)
                sg = self.gaussian2d(xx,yy,x0,y0,sig,sig,a)
                fl = sum(sg.flatten())
                gaus = np.pad(sg, ([min(xm),x.size - min(xm) + x.size],[min(ym),y.size - min(ym)+ym.size]))
                gaus = gaus[:x.size,:y.size]
                #gaus[xmask][:,ymask] += self.gaussian2d(xx,yy,x0,y0,sig,sig,a)
                # plt.imshow(gaus, norm=colors.LogNorm())
                # plt.scatter(y0,x0, color = 'r',alpha=0.5)
                # plt.show()
                self.img += gaus   
                coords.append(np.array([x0,y0]))
                swidths.append(np.array([sig,sig]))
                flux.append(fl)



        return {'coords': np.array(coords), 'widths': np.array(swidths), 'amps': amps, 'flux': flux}
        
    def gaussian2d(self, x, y, x0, y0, sigma_x, sigma_y, amplitude):
        exponent = -((x-x0)**2/(2*sigma_x**2) + (y-y0)**2/(2*sigma_y**2))
        return amplitude * np.exp(exponent)

    def sersic(self, x, y, x0, y0, Ie, Re, n, skew):
        R = np.sqrt(((x-x0))**2 + (((y - y0)/skew))**2)
        I = (Ie)*np.exp((-2*n - 1/3) * ((R/Re)**(1/n)))
        return I




    def createdataset(self, wdim = [64,64]):
        x = np.linspace(0,self.img.shape[1],self.img.shape[1])
        y = np.linspace(0,self.img.shape[0],self.img.shape[0])

        
        gwind = []
        for i, (coord, sig) in enumerate(zip(self.galprops['coords'],self.galprops['widths'])) :
            # rsize = (np.random.random(1)*8 + 3)
            winsamp = self.objwindow(x,y,coord[::-1], np.amax(sig),3,wdim)
            winsamp = (winsamp - np.amin(winsamp)) / np.amax(winsamp - np.amin(winsamp))
            gwind.append(winsamp)
        
        swind = []
        for i, (coord, sig) in enumerate(zip(self.starprops['coords'],self.starprops['widths'])):
            # rsize = (np.random.random(1)*8 + 3)
            winsamp = self.objwindow(x,y,coord[::-1], np.amax(sig),3,wdim)
            winsamp = (winsamp - np.amin(winsamp)) / np.amax(winsamp - np.amin(winsamp))
            swind.append(winsamp)
        
        gal = np.full(len(gwind),1)
        star = np.full(len(swind),0)

        allwinds = np.concatenate([gwind,swind])
        alllabels = np.concatenate([gal,star])
        print(alllabels)
        return {'label': alllabels, 'data': allwinds}

    def save_pkl(self, name):
        with open(f'{name}.pkl', 'wb') as f:
            pickle.dump(self,f)

    def show_img(self, img = None, objscatter = False):
        # plotting the image
        if img is None:
            img = self.img
        normalise = visualization.ImageNormalize(
            img,
            interval=visualization.AsymmetricPercentileInterval(5,92),
            stretch=visualization.LinearStretch()
            )
        fig, ax = plt.subplots(ncols=2,figsize=(10,8))
        ax[0].imshow(img,cmap='gnuplot')#, norm=colors.LogNorm())
        ax[1].imshow(img,cmap='Greys',norm=normalise)
        if objscatter:
            gcoords = self.galprops['coords']
            scoords = self.starprops['coords']
            ax[0].scatter(scoords[:,1],scoords[:,0],color = 'b', alpha = 0.5)
            ax[0].scatter(gcoords[:,1],gcoords[:,0],color = 'r', alpha = 0.5)
        #plt.show()

        return fig, ax

    def test_ident(self):
        cat = self.identify_objects(catalogue=True, bord_size = 1)
        flux = np.concatenate((self.galprops['flux'],self.starprops['flux']))
        peaks = np.concatenate(((self.galprops['amps'],self.starprops['amps'])))
        magnitude = self.zpinst - 2.5*np.log10(flux)
        print('yesna ', len(cat))
        cat = cat.dropna()
        #imgflux = 10**((self.zpinst - cat['magnitude_'])/2.5)
        print('Actual Mean: {}, Actual Std: {}'.format(magnitude.mean(),magnitude.std()))
        fig, ax = plt.subplots(3,1)
        
        magbinnum = int((max(magnitude) - min(magnitude))/0.2)
        n, bins, h1 = ax[0].hist(magnitude, bins = magbinnum, alpha = 0.5, label = '$\mu$: {:.3g}, $\sigma$ {:.3g}'.format(magnitude.mean(), magnitude.std()))
        datmagbinnum = int((max(cat['magnitude_']) - min(cat['magnitude_']))/0.2)
        n, bins, h2 = ax[0].hist(cat['magnitude_'], bins = datmagbinnum, alpha = 0.5, label = '$\mu$: {:.3g}, $\sigma$ {:.3g}'.format(cat['magnitude_'].mean(), cat['magnitude_'].std()))
        ax[0].legend()
        print('Detected Mean: {}, Detected Std: {}'.format(cat['magnitude_'].mean(),cat['magnitude_'].std()))
        ax[0].grid()
        print('num of objs', len(magnitude))
        print('detected objs:', len(cat['magnitude_']))
        ax[0].set_xlabel('Magnitude')
        

        peakbinnum = int((max(peaks) - min(peaks))/150)
        datpeakbinum = int((max(cat['peaks']) - min(cat['peaks']))/150)
        ax[1].hist(peaks, bins = peakbinnum, alpha = 0.5)
        ax[1].hist(cat['peaks'], bins = datpeakbinum, alpha = 0.5)
        ax[1].set_xlabel('Peak Intensity')
        ax[1].grid()
        ax[1].set_ylabel('Frequency')

        fluxbinnum = int((max(flux) - min(flux))/100000)
        datfluxbinum = int((max(cat['flux']) - min(cat['flux']))/100000)
        ax[2].hist(flux, bins = fluxbinnum, alpha = 0.5, label = '$\mu$: {:.3g}, $\sigma$ {:.3g}'.format(flux.mean(), flux.std()))
        ax[2].hist(cat['flux'], bins = datfluxbinum, alpha = 0.5, label = '$\mu$: {:.3g}, $\sigma$ {:.3g}'.format(cat['flux'].mean(), cat['flux'].std()))
        ax[2].set_xlabel('Flux')
        ax[2].grid()
        ax[2].legend()
         
        fig.legend((h1,h2), ('Actual Distribution','Detected Distribution'), loc = 'lower left')
        fig.tight_layout()
        plt.show()
        return cat
    
class Process_frompkl(Process_fake):


    def __init__(cls, pkl):
        with open('{}.pkl'.format(pkl), 'rb') as f:
            loaded = pickle.load(f)
        cls.zpinst = 25.3
        cls.zpinst_error = 0.02
        cls.noise1 = loaded.noise1
        cls.noise2 = loaded.noise2
        cls.img = loaded.img
        cls.galprops =  loaded.galprops
        cls.starprops = loaded.starprops
        cls.average_background = loaded.average_background
        cls.background_sigma = loaded.background_sigma
        












            

            


    
if __name__ == '__main__':        
    # to do list: 
    # 1) improve object detection - right now we detect most of it but not all of it
    # 2) find the local background and use that to find flux instead of global background
    # 3) errorbars

    testobjs = {'numgal': 3000, 'numstar': 1000, 'galamp': [15000,20], 'staramp': [10000,20], 'galwidth': [20,2], 'galskew': [1,2], 'gnrange': [0.5,4] , 'starwidth': [10,4]}
    test = Process_fake(size = [3000,3000], noise_params={'octaves': 2, 'seed': 55, 'scale': 17, 'shift': 3418.8155053212563, 'whiterange': 36, 'whiteshift': 5}, objparams=testobjs)

    test.save_pkl('TrainingData3')

    test.show_img(objscatter = True)

    #test = Process_frompkl('Trainingdata3')
    #test.img += np.random.normal(3450,100,(test.img.shape[0],test.img.shape[1]))

    fig, ax = plt.subplots(2,2)
    print(ax)
    cmap = 'inferno'
    normalise = visualization.ImageNormalize(test.img,interval=visualization.AsymmetricPercentileInterval(0,85),stretch=visualization.LinearStretch())  

    for i,ai in enumerate(ax):
        for j,aj in enumerate(ai):
            ax[i,j].tick_params(labelbottom = False, labelleft = False)

    im = ax[0,0].imshow(test.noise1, cmap = cmap)
    ax[0,0].set_xlabel('(a)')

    ax[0,1].imshow(test.noise2, cmap = cmap)
    ax[0,1].set_xlabel('(b)')

    cbar1 = ax[0,0].inset_axes([-0.2,0, 0.1, 1])
    ax[1,0].set_xlabel('(c)')
    fig.colorbar(im, cax = cbar1, location = 'left')

    im = ax[1,0].imshow(test.img, cmap = 'gnuplot2')#, norm = colors.LogNorm())
    ax[1,1].imshow(test.img, cmap = 'Greys', norm = normalise)
    ax[1,1].set_xlabel('(d)')

    cbar2 = ax[1,0].inset_axes([-0.2,0, 0.1, 1])
    fig.colorbar(im, cax = cbar2, location = 'left')
    #fig.tight_layout()
    #data = test.createdataset()
    plt.show()
    tcat = test.test_ident()

    x = tcat['x-centroid']
    y = tcat['y-centroid']

    plt.figure(figsize=(10,8))
    plt.imshow(test.img,cmap='Greys',norm=normalise)
    # plt.imshow(image.original,cmap='inferno',norm=colors.LogNorm())
    plt.plot(x,y,'.',color='red',alpha=0.5)
    for i in range(len(test.apertures)):
        test.apertures[i].plot(color='blue', lw=1.5, alpha=0.5)
    plt.colorbar()
    plt.show()

    # for i,(d,lab) in enumerate(zip(data['data'],data['label'])):
    #     print(lab)
    #     test.show_img(d)
    #     plt.show()