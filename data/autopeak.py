import numpy as np
import matplotlib.pyplot as plt # plotting package
import matplotlib.cm as cm # colormaps
import scipy.signal
import scipy.ndimage.filters
import os
import sys
import fourier
from scipy.optimize import curve_fit
from astropy.io import ascii
from astropy.table import Table
import glob
import re
import pandas as pd

def fnoise(x, a, b, c):
 return a/(x+b)+c

def clean(sequence,n,p):
#Task to iteratively clean a sequence of outliers at the n-sigma level, smoothing every p points.

  smoothseq = scipy.signal.medfilt(sequence, p)
  resid = sequence-smoothseq

# Do iterative sigma clipping of points beyond n-sigma:
  oldstd = scipy.stats.nanstd(resid)
  good = np.where(abs(resid) < n*scipy.stats.nanstd(resid))
  good = good[0]
  newstd = scipy.stats.nanstd(resid[good])
  while (1.-newstd/oldstd > 0.02):
    good1 = np.where(abs(resid[good]) < n*newstd)
    good = good[good1[0]]
    oldstd = newstd
    newstd = scipy.stats.nanstd(resid[good])

  return good

#peak finding algorithm - in autocorrelation
def count(times,values,minamp):
  dtimes=[times[0]]
  zpoints=[values[0]]

  first=(np.where(np.abs(values-values[0]) > minamp))[0]

  if (len(first) != 0):
     dtimes = np.append(dtimes,times[first[0]])
     zpoints = np.append(zpoints,values[first[0]])

     flip=int(np.abs(zpoints[1]-zpoints[0])/(zpoints[1]-zpoints[0]))

     y0=zpoints[0]
     y1=zpoints[1]

     for i in np.arange(first[0]+1,len(times)):
        if (flip == 1):
         if ((values[i]-y0) > (y1-y0)):
           zpoints[len(zpoints)-1]=values[i]
           dtimes[len(zpoints)-1]=times[i]
           y1=values[i]

         if ((y1-values[i]) > minamp):
           zpoints = np.append(zpoints,values[i])
           dtimes = np.append(dtimes,times[i])
           y0=y1
           y1=values[i]
           flip=-1

        if (flip == -1):
         if ((y0-values[i]) > (y0-y1)):
           zpoints[len(zpoints)-1]=values[i]
           dtimes[len(zpoints)-1]=times[i]
           y1=values[i]

         if ((values[i]-y1) > minamp):
           zpoints=np.append(zpoints,values[i])
           dtimes=np.append(dtimes,times[i])
           y0=y1
           y1 = values[i]
           flip = 1

  result = np.array([dtimes,zpoints])

  return result

#use ticid for star
def autopeak(time,flux2,star,sector):

# remove 7-sigma outliers
  time = time[flux2>0]
  flux2 = flux2[flux2>0]
  avg = np.nanmedian(-2.5*np.log10(flux2))
  rms = np.nanstd(-2.5*np.log10(flux2))

  fine = np.where((abs(-2.5*np.log10(flux2)-avg) < 7.*rms) & (~np.isnan(flux2)))
  fine = fine[0]

  avg = np.median(-2.5*np.log10(flux2[fine]))
  rms = np.nanstd(-2.5*np.log10(flux2[fine]))

  mag = -2.5*np.log10(flux2[fine])

  #FIGURE THIS PART OUT  
  if fine.size==0:
    result = np.array([0,1,0,0,1,0])
    return result

# Interpolate onto evenly spaced grid...
  ndatapoints=7040.
  ndays=(time[fine])[len(time[fine])-1]-time[fine[0]]
  u=ndatapoints/ndays

  inttime=(np.arange(ndatapoints,dtype='float')/u)+(time[fine])[0]-(time[fine])[0]
  maginterp=np.interp(inttime,time[fine]-(time[fine])[0],mag)

#  Compute autocorrelation function:
  lag = np.arange(len(maginterp),dtype='float')
  unbias = maginterp-np.mean(maginterp)
  norm = np.sum(unbias**2)    

  ac = np.correlate(unbias,unbias, "full")/norm
  ac = ac[round(len(ac)/2)-1:]

  result=count(lag/u,ac,0.05)
  autopeaks = (result[1])[np.arange(round((len(result[0])+1)/2)-1)*2]
  autopeaktimes = (result[0])[np.arange(round((len(result[0])+1)/2)-1)*2]

  autopeaks = autopeaks[1:len(autopeaks)]
  autopeaktimes = autopeaktimes[1:len(autopeaktimes)]
  
  if (len(autopeaks)>=1):
    bigpeak=np.max(autopeaks)
    bigpeaklag = np.argmax(autopeaks)
    finalper = autopeaktimes[bigpeaklag]
  else:
    finalper = 0.00000000001

  #potential peaks in the autocorrelation functino (plots red dots if exist)
  periodicity = 0
  if (len(autopeaks) >= 1):
   acfit,accov = curve_fit(fnoise,lag/u,ac)
   acsmoothed = acfit[0]/(lag/u+acfit[1])+acfit[2]
   x= np.where(ac == bigpeak)
   x= x[0][0].item()
   peaksmooth = acsmoothed[x]
   periodicity = bigpeak - peaksmooth

   if (len(autopeaks) > 1):
        
  #use bigpeak, then fit slowly varying curve to the autocorrelation (ac)
     if (bigpeak > 1.5*np.max(autopeaks[0:2])):
       finper = finalper
       
     else:
       
       finper = 0
  
  try:  
    if (autopeaks[0]>autopeaks[1]):
        result = 1
    elif (autopeaks[0]<autopeaks[1]):
        result = 2
  except:
    result = 0
  return result

def agraph(tic, remove_outliers = 0):
    row = ref[ref.TIC_ID.isin([tic])]
    string = str(row.Sector)
    s = string.split(" ")
    files = data_dir + ref.Filename.values
    filepath = files[int(s[0])]
    lightc = loaders.load_lc(filepath)
    tic = ref.TIC_ID.values
    tic = tic[int(s[0])]
    sec = ref.Sector.values
    sec = sec[int(s[0])]
    a = autopeak(lightc.time.value,lightc.flux.value,tic,sec)
    #print(a)
    return a

def agraph2(tic, remove_outliers = 0):
    row = ref[ref.TIC_ID.isin([tic])]
    string = str(row.Sector)
    s = string.split(" ")
    files = data_dir + ref.Filename.values
    filepath = files[int(s[0])]
    lightc = loaders.load_lc(filepath)
    tic = ref.TIC_ID.values
    tic = tic[int(s[0])]
    sec = ref.Sector.values
    sec = sec[int(s[0])]
    a = autopeak2(lightc.time.value,lightc.flux.value,tic,sec)
    #print(a)
    return a
