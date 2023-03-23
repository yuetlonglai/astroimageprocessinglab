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


class Process:

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

    def circle_lowpass(self, cutoff, tau, hp = False):
        # self.show_img(self.img)
        circ = self.draw_circle(cutoff, tau)

        imgfft = fft.fft2(self.img)
        imgfft = fft.fftshift(imgfft)
        # self.show_img(abs(imgfft))
        # print(imgfft)

        imgmask = np.multiply(imgfft, circ[:imgfft.shape[0],:imgfft.shape[1]]) 

        # self.show_img(circ)
        # self.show_img(abs(imgmask))

        imgifft = fft.ifft2(imgmask)
        if hp:
            hp_mask = np.multiply(imgfft,1 - circ[:imgfft.shape[0],:imgfft.shape[1]])
            imghp = fft.ifft2(hp_mask)

            return (abs(imgifft), abs(imghp))


        return abs(imgifft)


    def draw_circle(self, r, tau):
        shape = np.asarray([int(i/2) for i in self.img.shape]) + [1,0]
        canv = np.full(shape, 0, dtype = float)
        center = np.array(self.img.shape, dtype = int)/2.0
        min_dist = 1e8 
        for i in tqdm(range((len(canv)))):
            for j in range((len(canv[i]))):
                dist = (i- center[0])**2 + (j - center[1])**2
                if  dist <= r ** 2:
                    canv[i,j] = 1
                else:
                    #val = -tau*np.sqrt(dist - r**2) + 1
                    val = np.exp(-(tau*(np.sqrt(dist - r**2))))
                    if val < 0:
                        val = 0
                    canv[i,j] = val
        
        canv = np.pad(canv,((0,shape[0]),(0,shape[1])), mode = 'reflect')
        return canv
    
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

    def identify_objects(self,edge=150,boxedge=False): # important method because it dictates the quality of the results
        # set threshold of the magnitude of objects
        # thresh = 5e3
        thresh =  self.background_sigma / 2
        image_no_edge = self.mask_edge(edge)
        bw = morphology.closing(image_no_edge > thresh, morphology.disk(1))
        # remove artifacts connected to image border
        cleared = segmentation.clear_border(bw)
        # label image regions
        label_image = measure.label(cleared)
        spotsx = []
        spotsy = []
        edgex = []
        edgey = []
        for region in measure.regionprops(label_image):
            # take regions with large enough areas
            if region.area >= np.pi*2**2:
                # draw rectangle around segmented coins
                minr, minc, maxr, maxc = region.bbox
                y = int((minr+maxr)/2)
                x = int((minc+maxc)/2)
                if self.img[y][x] > thresh and self.img[y+1][x] > thresh and self.img[y-1][x] > thresh and self.img[y][x+1] > thresh and self.img[y][x-1] > thresh:
                    spotsy.append(y)
                    spotsx.append(x)
                    edgey.append((minr, minr, maxr, maxr, minr))
                    edgex.append((minc, maxc, maxc, minc, minc))
        print('Number of Object Detected : ' + str(len(spotsx)))
        if boxedge == False:
            return spotsy, spotsx
        else:
            return spotsy, spotsx, edgey, edgex
    
    def aperture_photometry(self,radius=6):
        y, x = self.identify_objects()
        self.fluxes = []
        for i in range(len(x)): # scanning each detected object's total flux radially in both x and y directions 
            flux = 0
            for k in range(-radius,radius+1): # vertical
                for j in range(-(radius-abs(k)),radius-abs(k)): # horizontal
                    flux += self.img[y[i]+k][x[i]+j] - self.average_background
            self.fluxes.append(flux) # assume constant background contribution for now
        # calibrating the fluxes using ZPinsturment values from header
        self.fluxes = np.array(self.fluxes)
        self.fluxes = self.zpinst - 2.5 * np.log10(self.fluxes) # value
        self.fluxes_error = self.zpinst_error - 2.5 * np.log10(self.fluxes) # value error
        return self.fluxes, self.fluxes_error
    
    def catalogue(self):
        y, x = self.identify_objects()
        z, zerr = self.aperture_photometry()
        self.table = pd.DataFrame(columns=['position-x','position-y','magnitude','magnitude-error'], dtype = object)
        self.table['position-x'] = x
        self.table['position-y'] = y
        self.table['magnitude'] = z
        self.table['magnitude-error'] = zerr
        return self.table
    
    def number_count(self, plotting=False):
        cat = self.catalogue()
        m = np.linspace(min(cat['magnitude']),max(cat['magnitude']),50)
        N = []
        Nerr_l = []
        Nerr_u = []
        for i in m:
            N.append(np.count_nonzero(np.where(cat['magnitude'] < i, True, False)))
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
            plt.errorbar(x=m,y=np.log10(N),fmt='.',color='blue',label='Data')
            plt.plot(m,logN(m),'-',color='red',label='Fitted')
            plt.legend(loc='best')
            plt.show()
        return m, N#, abs(np.array(N)-np.array(Nerr_l)), abs(np.array(Nerr_u)-np.array(N))

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

    def double_lowpass(self, hifrq, lowfreq):
        lp = self.circle_lowpass(1600, 0.01)
        
        lp2 = self.circle_lowpass(3,1)
        self.show_img(lp2)
        self.img = lp - lp2


class Process_fake(Process):
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
        wnoise = np.random.rand(size[0],size[1])*noise_params['whiterange'] + noise_params['whiteshift']
        pic += wnoise
        

        #pickles = glob('*backpickle*')
        #print(pickles)
        #check = [str(noise_params['seed']) not in (i) or (str(size[0]) + '_' + str(size[1])) not in (i) for i in pickles]
        # print(check)
        # if all(check):
        #     noise = PerlinNoise(octaves = noise_params['octaves'], seed = noise_params['seed'])
        #     pic = np.zeros(size)
        #     for i in tqdm(range(size[0])):
        #         for j in range(size[1]):
        #             pic[i,j] = (noise([i/size[0], j/size[1]]))*noise_params['scale'] + noise_params['shift']
        #     with open('{}_{}_backpickle_{}.pkl'.format(size[0],size[1],noise_params['seed']), 'wb') as f:
        #         pickle.dump(pic,f)
        
        # else:
        #     with open('{}_{}_backpickle_{}.pkl'.format(size[0],size[1],noise_params['seed']), 'rb') as f:
        #         pic = pickle.load(f)
            
        
        
        cls.img = pic
        print(cls.img)
        print(cls.img.shape)
        cls.show_img()
        cls.galprops =  cls.rand_galaxy(objparams['numgal'],objparams['galamp'],objparams['galwidth'], objparams['galskew'], objparams['gnrange'])
        cls.starprops = cls.rand_star(objparams['numstar'],objparams['staramp'],objparams['starwidth'])
        cls.average_background = noise_params['shift']
        cls.background_sigma = noise_params['scale']/2
        cls.show_img()


    def rand_galaxy(self, num, ampr, widthr, skew, nrange):
        xpoints = np.random.randint(len(self.img[0]),size = num)
        ypoints = np.random.randint(len(self.img[:,0]),size = num)

        xrs = np.random.randint(min(widthr),max(widthr),size = num)
        skews = abs(np.random.rand(num))*abs(skew[0]-skew[1]) + min(skew)
        amps = np.random.randint(min(ampr),max(ampr), size = num)

        x = np.arange(0,len(self.img[0]),1, dtype=int)
        y = np.arange(0,len(self.img[1]),1, dtype=int)

        ns = np.random.rand(num) * (max(nrange) - min(nrange)) + min(nrange)

        
        coords = []
        swidths = []
        for i, (x0,y0,xr,s,n,a) in tqdm(enumerate(zip(xpoints,ypoints,xrs,skews,ns,amps))):
            
            angle = np.random.randint(0,360,1)

            xmask = (x < x0 + 4*xr) & (x > x0 - 4*xr)
            ymask = (y0 - 4*xr < y) & (y < y0 + 4*xr)

            xm = x[xmask]
            ym = y[ymask]

            if xm.size == ym.size:

                xx, yy = np.meshgrid(xm,ym)
                sg = self.sersic(xx,yy,x0,y0,a,xr,n,s)
                
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

                self.img += np.array(gimg)

        return {'coords': np.array(coords), 'widths': np.array(swidths), 'amps': amps}

    def rand_star(self, num, maxamp, maxwidth):
        xpoints = np.random.randint(self.img.shape[0],size = num)
        ypoints = np.random.randint(self.img.shape[0],size = num)
        
        widths = np.random.randint(min(maxwidth),max(maxwidth),size = num)

        amps = np.random.randint(min(maxamp),max(maxamp), size = num)

        x = np.arange(0,len(self.img[0]),1, dtype=int)
        y = np.arange(0,len(self.img[1]),1, dtype=int)

        coords = []
        swidths = []
        for i, (x0,y0,sig,a) in tqdm(enumerate(zip(xpoints,ypoints,widths,amps))):

            xmask = (x < x0 + 4*sig) & (x > x0 - 4*sig)
            ymask = (y0 - 4*sig < y) & (y < y0 + 4*sig)

            xm = x[xmask]
            ym = y[ymask]

            if ym.size == xm.size: 

                xx, yy = np.meshgrid(xm,ym)
                sg = self.gaussian2d(xx,yy,x0,y0,sig,sig,a)
                gaus = np.pad(sg, ([min(xm),x.size - min(xm) + x.size],[min(ym),y.size - min(ym)+ym.size]))
                gaus = gaus[:x.size,:y.size]
                #gaus[xmask][:,ymask] += self.gaussian2d(xx,yy,x0,y0,sig,sig,a)
                # plt.imshow(gaus, norm=colors.LogNorm())
                # plt.scatter(y0,x0, color = 'r',alpha=0.5)
                # plt.show()
                self.img += gaus   
                coords.append(np.array([x0,y0]))
                swidths.append(np.array([sig,sig]))


        return {'coords': np.array(coords), 'widths': np.array(swidths), 'amps': amps}
        
    def gaussian2d(self, x, y, x0, y0, sigma_x, sigma_y, amplitude):
        exponent = -((x-x0)**2/(2*sigma_x**2) + (y-y0)**2/(2*sigma_y**2))
        return amplitude * np.exp(exponent)

    def sersic(self, x, y, x0, y0, Ie, Re, n, skew):
        R = np.sqrt((x-x0)**2 + ((y - y0)/skew)**2)
        I = (Ie)*np.exp((-2*n - 1/3) * ((R/Re)**(1/n)))
        return I
    
    def objwindow(self, x, y, coords, rng, wscale, wdim):
        xmask = (coords[0] - wscale* rng < x) & (x < coords[0] +  wscale* rng)
        ymask = (coords[1] - wscale * rng < y) & (y < coords[1] + wscale * rng)


        window = self.img[xmask][:,ymask]

        winterp = RegularGridInterpolator((x[xmask],y[ymask]),window)

        nx = np.linspace(min(x[xmask]),max(x[xmask]),wdim[0])
        ny = np.linspace(min(y[ymask]),max(y[ymask]),wdim[1])

        NX, NY = np.meshgrid(nx,ny)

        newdat = np.reshape(winterp((NX,NY)),(64,64,1))
        return newdat




    def createdataset(self, wdim = [64,64]):
        x = np.linspace(0,self.img.shape[0],self.img.shape[0])
        y = np.linspace(0,self.img.shape[1],self.img.shape[1])

        
        gwind = []
        for i, (coord, sig) in enumerate(zip(self.galprops['coords'],self.galprops['widths'])) :
            winsamp = self.objwindow(x,y,coord, np.amax(sig),2,wdim)
            winsamp = (winsamp - np.amin(winsamp)) / np.amax(winsamp - np.amin(winsamp))
            gwind.append(winsamp)
        
        swind = []
        for i, (coord, sig) in enumerate(zip(self.starprops['coords'],self.starprops['widths'])):
            rsize = (np.random.random(1)*8 + 3)
            winsamp = self.objwindow(x,y,coord, np.amax(sig),rsize,wdim)
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
            interval=visualization.AsymmetricPercentileInterval(5,95),
            stretch=visualization.LinearStretch()
            )
        fig, ax = plt.subplots(ncols=2,figsize=(10,8))
        ax[0].imshow(img,cmap='inferno', norm=colors.LogNorm())
        ax[1].imshow(img,cmap='Greys',norm=normalise)
        if objscatter:
            gcoords = self.galprops['coords']
            scoords = self.starprops['coords']
            ax[0].scatter(scoords[:,1],scoords[:,0],color = 'b', alpha = 0.5)
            ax[0].scatter(gcoords[:,1],gcoords[:,0],color = 'r', alpha = 0.5)
        plt.show()



            

            


    
if __name__ == '__main__':        
    # to do list: 
    # 1) improve object detection - right now we detect most of it but not all of it
    # 2) find the local background and use that to find flux instead of global background
    # 3) errorbars

    # testobjs = {'numgal': 2000, 'numstar': 2000, 'galamp': [500,5000], 'staramp': [5000,500], 'galwidth': [20,2], 'galskew': [1,2], 'gnrange': [0.5,4] , 'starwidth': [10,2]}
    # test = Process_fake(size = [3000,3000], noise_params={'octaves': 2, 'seed': 55, 'scale': 17, 'shift': 3418.8155053212563, 'whiterange': 5*36, 'whiteshift': 5}, objparams=testobjs)

    # test.save_pkl('2000gal_2000star_3000_3000')

    # test.show_img(objscatter = True)

    with open('2000gal_2000star_3000_3000.pkl', 'rb') as f:
        test = pickle.load(f)

    test.show_img(objscatter = True)

    data = test.createdataset()


    for i,(d,lab) in enumerate(zip(data['data'],data['label'])):
        print(lab)
        test.show_img(d)
        plt.show()

    #using the class
    #image = Process('A1_mosaic.fits')

    #bleeding = [(1400,1438),(514,558),(578,1456),(851,2133),(1292,776),(1196,2465),(1838,973),(2325,905),(2301,2131),(3184,2088),(4602,1431),(4488,1531)]
    #image.clean_bleeding(bleeding,clean_background=False)

    #image = test

    #lp, hp = image.circle_lowpass(2000, 0.5, True)

    #image.show_img(hp)
    #print(np.std(hp), np.mean(hp))
    #image.double_lowpass()

    # y,x,by,bx = image.identify_objects(boxedge=True)
    # plt.figure(figsize=(10,8))
    # normalise = visualization.ImageNormalize(image.img,interval=visualization.AsymmetricPercentileInterval(5,95),stretch=visualization.LinearStretch())
    # plt.imshow(image.img,cmap='Greys',norm=normalise)
    # plt.plot(x,y,'.',color='red',alpha=0.5)
    # for i in range(len(bx)):
    #     plt.plot(bx[i],by[i],'-',color='blue',linewidth=1)
    # plt.colorbar()
    # plt.show()

    # #Number of Count Plot
    # image.number_count(plotting=True)
            