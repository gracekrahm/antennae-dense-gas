import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
from astropy.io import fits
from astropy import stats, units as u
from astropy.stats import mad_std
from astropy.coordinates import SkyCoord
import csv
import pandas as pd
from astrodendro import Dendrogram
import peakutils.peak
import datetime
import multiprocessing as mp
import sys
import astropy.wcs

import warnings
warnings.simplefilter('ignore')



class DataCubes:
        def __init__(self,cubefile):
                data = fits.getdata(cubefile)
                hdr = fits.getheader(cubefile)
                restfreq = (hdr['RESTFRQ']*u.Hz
                rms = rms = stats.mad_std(data[~np.isnan(data)])

                bmaj = hdr['BMAJ']*3600*u.arcsec
                bmin = hdr['BMIN']*3600*u.arcsec
                if hdr['CUNIT3']=='m/s':
                        freq = hdr['CRVAL3']*u.m/u.s
                        freqGHz = freq.to(u.GHz, equivalencies=u.doppler_radio(restfreq))
                        self.deltav = abs(hdr['cdelt3'])/1000. * u.km / u.s
                elif hdr['CUNIT3']=='Hz':
                        freqGHz = hdr['CRVAL3']*u.GHz/10**9
                        self.deltav = 2.99792458e5 * np.absolute(dnu=hdr['cdelt3'])/restfreq * u.km / u.s
                self.freq = freqGHz

                #possibly unncessary to have for each cube since they're smoothed but just in case
                self.cdelt2 = hdr['cdelt2']*3600*u.arcsec #arcsec per pixel
                self.brad = np.sqrt(bmaj*bmin)/206265. * D_Gal
                self.pixelarea = ((self.cdelt2.value/206265.) * D_Gal * pc)**2 # cm^2


                #convert from jy/beam to K
                beam_area = 2*np.pi*bmaj*bmin
                self.data = (data*u.Jy/beam_area).to(u.K, equivalencies=u.brightness_temperature(self.freq))
                self.rms = rms.to(u.K, equivalencies=u.brightness_temperature(self.freq))




def fitEllipse(cont):
    # From online stackoverflow thread about fitting ellipses
        x=cont[:,0]
        y=cont[:,1]

        x=x[:,None]
        y=y[:,None]

        D=np.hstack([x*x,x*y,y*y,x,y,np.ones(x.shape)])
        S=np.dot(D.T,D)
        C=np.zeros([6,6])
        C[0,2]=C[2,0]=2
        C[1,1]=-1
        E,V=np.linalg.eig(np.dot(np.linalg.inv(S),C))
        n=np.argmax(E)
        a=V[:,n]

    #-------------------Fit ellipse-------------------
        b,c,d,f,g,a=a[1]/2., a[2], a[3]/2., a[4]/2., a[5], a[0]
        num=b*b-a*c
        cx=(c*d-b*f)/num
        cy=(a*f-b*d)/num

        angle=0.5*np.arctan(2*b/(a-c))*180/np.pi
        up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
        down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
        down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
         a=np.sqrt(abs(up/down1))
        b=np.sqrt(abs(up/down2))

        params=[cx,cy,a,b,angle]

        return params

def define_get_clump_props(stype, clumps, HCO+, HCN, CO, nsig, D_Gal,PLOT=False, verbose=False):
    # d can be either a dendrogram file or a clumpfind array, stype (structure type) will tell the code which it is
    global get_clump_props
    def get_clump_props(ncl):
        print('Computing Clump ', ncl, datetime.datetime.now())
        if stype=='dendro':
            if clumps[ncl].parent == None and clumps[ncl].children!=None:
                cltype = 0
            elif clumps[ncl].is_leaf:
                cltype = 2
            elif clumps[ncl].is_branch:
                cltype = 1
            else:
                cltype = 0
            mask = clumps[ncl].get_mask()
                            
        else:
            cldat = np.where(clumps==ncl, TCO, np.nan)
            cltype = 2
                            
        if np.all(np.isnan(cldat)):
        blank = np.full(22, np.nan)
        blank[0] = ncl
        print('clump ',ncl, ' is all nans')
        print
        return blank                            
        
        ##calculations                     
        props = [ncl, cltype, argmax[2], argmax[1], argmax[0], radec,SGMC, Npix, Nvox, lumco.value, errlumco, COmax.value, mlumco.value, errmlumco, line_ratio,alphaco,meansigv, errsigv, a.value, b.value, R, errR.value, area.value, perim.value, density,densityerr,pressure, pressure_err,alphavir,erralphavir]
        proplist = []
        for i in props:
            if type(i) == type(lumco.value):
                #i = format(i,'.4f')
                i="{0:.4g}".format(i).rstrip('0').rstrip('.')
            proplist.append(i)
        final_props = np.array(proplist)
        print('density,density err', density,densityerr)
        print('alphavir', alphavir)
        return final_props


    return get_clump_props         
                            
                            
                            
                            
                            


if __name__=='__main__':
        #changables
        SAVE = True
        PLOT = False
        TEST = True
        verbose = True
        nsig = 5 # Typically use a cut of 5 sigma in maps
        D_Gal = 22*10**6 * u.pc
        deconvolve = True
        xradius = True
        stype = 'clump'
        maskfile = 'ab612co21.clumps.6sig350npix.fits'
        dir = '/lustre/cv/students/gkrahm/ant_dense/files/final_cubes/'
        cubefilehcn = dir + 'hcn.reshaped.pbcor.fits'
        cubefilehco = dir + 'hco.reshaped.pbcor.fits'
        cubefileco = dir + 'co21.reshaped.pbcor.fits'



        asarea = ((u.arcsec*D_Gal)**2).to(u.pc**2,equivalencies=u.dimensionless_angles())

        print('Running clump analysis for Antennae')
        print('Loading files', datetime.datetime.now())
        if stype == 'clump':
                clumps = fits.getdata(maskfile) # Standard clumpfind output fits
                clmax = np.nanmax(clumps)
        else:
                clumps = Dendrogram.load_from(maskfile)
                clmax = len(clumps)
        HCO+ = DataCubes(cubefilehco)
        HCN = DataCubes(cubefilehcn)
        CO = DataCubes(cubefileco)



        vels = nnp.flip(np.arange(1794.5669772, HCN.deltav*899, HCN.deltav))


        get_clump_props = (stype, clumps, HCO+, HCN, CO, nsig, D_Gal, PLOT=PLOT, verbose=verbose)
        print('Function defined', datetime.datetime.now())


        if TEST:
                props_array = []
                for ncl in range(1,3):
                        props = get_clump_props(ncl)
                        props_array.append(props)
                        print('Done clump', ncl, datetime.datetime.now())

        else:
                print('Starting mp clump props for ', clmax, ' structures.', datetime.datetime.now())
                print
                pool = mp.Pool(processes=procs)
                if stype=='clump':
                        props_array = np.array(pool.map(get_clump_props, np.arange(1,clmax+1)))
                else:
                        props_array = np.array(pool.map(get_clump_props, np.arange(clmax)))
                pool.close()
                pool.join()

                print
                print('Done parallel computing', datetime.datetime.now())
        print(props_array.shape)
