#import and set up everything
import pandas as pd
import numpy as np
import sys
sys.path.append("../PhaseFold")
import os
import data
from data import loaders
import lightkurve as lk
import scipy.signal
import matplotlib.pyplot as plt
import math
from PIL import Image
import warnings
from ipywidgets.widgets import Button, Layout
from IPython.display import display
import functools

warnings.filterwarnings("ignore", category=RuntimeWarning) 

#before using the functions, create a folder called LightCurves
def foldperiod(lcurve, sect, givenperiod):
    lightc = lcurve
    a = lightc.scatter()
    lc = lightc[lightc.quality==0]
    pg = lc.normalize(unit='ppm').to_periodogram(minimum_period = 0.042, oversample_factor=300)
    period = pg.period_at_max_power
    pg1 = lc.normalize(unit='ppm').to_periodogram(maximum_period = 2.1*period.value, oversample_factor=100)
    b = pg1.plot(view='period')
    folded = lc.fold(givenperiod)
    c = folded.scatter(label=fr'Period = {givenperiod:.5f} d')
    
    #creates the save button that can be used to save the images, as well as the combined version
    b_save = Button (description = 'save', layout = Layout(width='100px'))
    def bsave(b_save):
        a.figure.savefig(f'LightCurves/S{sect}TIC{lc.TARGETID}/S{sect}TIC{lc.TARGETID}_LC.png')
        b.figure.savefig(f'LightCurves/S{sect}TIC{lc.TARGETID}/S{sect}TIC{lc.TARGETID}_Periodogram.png')
        c.figure.savefig(f'LightCurves/S{sect}TIC{lc.TARGETID}/S{sect}TIC{lc.TARGETID}_Folded.png')
        images = [Image.open(x) for x in [f'./LightCurves/S{sect}TIC{lc.TARGETID}/S{sect}TIC{lc.TARGETID}_LC.png', f'./LightCurves/S{sect}TIC{lc.TARGETID}/S{sect}TIC{lc.TARGETID}_Periodogram.png', f'./LightCurves/S{sect}TIC{lc.TARGETID}/S{sect}TIC{lc.TARGETID}_Folded.png']]
        widths, heights = zip(*(i.size for i in images))
        total_height = sum(heights)
        min_width = min(widths)
        new_im = Image.new('RGB', (min_width, total_height))
        y_offset = 0
        for im in images:
            new_im.paste(im, (0,y_offset))
            y_offset += im.size[1]
        new_im.save(f'./LightCurves/S{sect}TIC{lc.TARGETID}/S{sect}TIC{lc.TARGETID}.png')
    b_save.on_click(bsave)
    display(b_save)

#prints out nothing, outputs the light curve times and fluxes    
def calcfold(lcurve, sect):
    '''
    Phase folds the given light curve and outputs 3 graphs: the original light curve, the periodogram, and the folded light curve
    As well as gives the option to save the 3 graphs and the combined graph in the LightCurves folder.
    
        Parameters:
            lcurve (lightkurve.LightCurve): a lightkurve.LightCurve object
            sect (int): the sector that the light curve is in
    '''
    lightc = lcurve
    lc = lightc[lightc.quality==0]
    pg = lc.normalize(unit='ppm').to_periodogram(minimum_period = 0.042, oversample_factor=300)
    period = pg.period_at_max_power
    pg1 = lc.normalize(unit='ppm').to_periodogram(maximum_period = 2.1*period.value, oversample_factor=100)
    mini = .7*period
    maxi = 1.3*period
    midpt = (mini + maxi)/2
    midpt = redef(mini, maxi, midpt, lc)
    folded = lc.fold(midpt)
    return midpt,folded

def fold(lcurve, sect, remove_outliers = 0):
    '''
    Phase folds the given light curve and outputs 3 graphs: the original light curve, the periodogram, and the folded light curve
    As well as gives the option to save the 3 graphs and the combined graph in the LightCurves folder.
    
        Parameters:
            lcurve (lightkurve.LightCurve): a lightkurve.LightCurve object
            sect (int): the sector that the light curve is in
    '''
    lightc = lcurve
    if remove_outliers == 0:
        a = lightc.scatter()
    else:
        a = lightc.remove_outliers(sigma_lower = 7, sigma_upper = 3).scatter()
    lc = lightc[lightc.quality==0]
    pg = lc.normalize(unit='ppm').to_periodogram(minimum_period = 0.042, oversample_factor=300)
    period = pg.period_at_max_power
    pg1 = lc.normalize(unit='ppm').to_periodogram(maximum_period = 2.1*period.value, oversample_factor=100)
    mini = .7*period
    maxi = 1.3*period
    midpt = (mini + maxi)/2
    b = pg1.plot(view='period')
    midpt = redef(mini, maxi, midpt, lc)
    folded = lc.fold(midpt)
    #cleanlightcurve = folded[folded.quality==0]
    c = folded.scatter(label=fr'Period = {midpt.value:.5f} d')
    
    #creates the save button that can be used to save the images, as well as the combined version
    b_save = Button (description = 'save', layout = Layout(width='100px'))
    def bsave(b_save):
        a.figure.savefig(f'LightCurves/S{sect}TIC{lc.TARGETID}/S{sect}TIC{lc.TARGETID}_LC.png')
        b.figure.savefig(f'LightCurves/S{sect}TIC{lc.TARGETID}/S{sect}TIC{lc.TARGETID}_Periodogram.png')
        c.figure.savefig(f'LightCurves/S{sect}TIC{lc.TARGETID}/S{sect}TIC{lc.TARGETID}_Folded.png')
        images = [Image.open(x) for x in [f'./LightCurves/S{sect}TIC{lc.TARGETID}/S{sect}TIC{lc.TARGETID}_LC.png', f'./LightCurves/S{sect}TIC{lc.TARGETID}/S{sect}TIC{lc.TARGETID}_Periodogram.png', f'./LightCurves/S{sect}TIC{lc.TARGETID}/S{sect}TIC{lc.TARGETID}_Folded.png']]
        widths, heights = zip(*(i.size for i in images))
        total_height = sum(heights)
        min_width = min(widths)
        new_im = Image.new('RGB', (min_width, total_height))
        y_offset = 0
        for im in images:
            new_im.paste(im, (0,y_offset))
            y_offset += im.size[1]
        new_im.save(f'./LightCurves/S{sect}TIC{lc.TARGETID}/S{sect}TIC{lc.TARGETID}.png')
    b_save.on_click(bsave)
    display(b_save)

def fold2(lcurve, sect, lbound, ubound, lbound2, ubound2):
    '''
    Phase folds the given light curve and outputs 3 graphs: the original light curve, the periodogram, and the folded light curve
    As well as gives the option to save the 3 graphs and the combined graph in the LightCurves folder.
    
        Parameters:
            lcurve (lightkurve.LightCurve): a lightkurve.LightCurve object
            sect (int): the sector that the light curve is in
    '''
    lightc = lcurve
    a = lightc.scatter()
    a.set_ylim(lbound,ubound)
    lc = lightc[lightc.quality==0]
    pg = lc.normalize(unit='ppm').to_periodogram(minimum_period = 0.042, oversample_factor=300)
    period = pg.period_at_max_power
    pg1 = lc.normalize(unit='ppm').to_periodogram(maximum_period = 2.1*period.value, oversample_factor=100)
    mini = .7*period
    maxi = 1.3*period
    midpt = (mini + maxi)/2
    b = pg1.plot(view='period')
    midpt = redef(mini, maxi, midpt, lc)
    folded = lc.fold(midpt)
    #cleanlightcurve = folded[folded.quality==0]
    c = folded.scatter(label=fr'Period = {midpt.value:.5f} d')
    c.set_ylim(lbound2,ubound2)
    
    #creates the save button that can be used to save the images, as well as the combined version
    b_save = Button (description = 'save', layout = Layout(width='100px'))
    def bsave(b_save):
        a.figure.savefig(f'LightCurves/S{sect}TIC{lc.TARGETID}/S{sect}TIC{lc.TARGETID}_LC.png')
        b.figure.savefig(f'LightCurves/S{sect}TIC{lc.TARGETID}/S{sect}TIC{lc.TARGETID}_Periodogram.png')
        c.figure.savefig(f'LightCurves/S{sect}TIC{lc.TARGETID}/S{sect}TIC{lc.TARGETID}_Folded.png')
        images = [Image.open(x) for x in [f'./LightCurves/S{sect}TIC{lc.TARGETID}/S{sect}TIC{lc.TARGETID}_LC.png', f'./LightCurves/S{sect}TIC{lc.TARGETID}/S{sect}TIC{lc.TARGETID}_Periodogram.png', f'./LightCurves/S{sect}TIC{lc.TARGETID}/S{sect}TIC{lc.TARGETID}_Folded.png']]
        widths, heights = zip(*(i.size for i in images))
        total_height = sum(heights)
        min_width = min(widths)
        new_im = Image.new('RGB', (min_width, total_height))
        y_offset = 0
        for im in images:
            new_im.paste(im, (0,y_offset))
            y_offset += im.size[1]
        new_im.save(f'./LightCurves/S{sect}TIC{lc.TARGETID}/S{sect}TIC{lc.TARGETID}.png')
    b_save.on_click(bsave)
    display(b_save)


# Folds and saves the original light curve, periodogram, folded light curve, and all 3 combined
# Into a folder in the LightCurves folder
def foldandsave(lcurve, sect):
    '''
    Folds and saves the original light curve, periodogram, folded light curve, and all 3 combined
    Into a folder in the LightCurves folder
    
        Parameters:
            lcurve (lightkurve.LightCurve): a lightkurve.LightCurve object
            sect (int): the sector that the light curve is in
    '''
    lightc = lcurve
    a = lightc.scatter()
    lc = lightc[lightc.quality==0]
    path = 'LightCurves/'
    isExist = os.path.exists(path + f'S{sect}TIC{lc.TARGETID}')
    if not isExist:
        os.makedirs(path + f'S{sect}TIC{lc.TARGETID}')
    a.figure.savefig(f'LightCurves/S{sect}TIC{lc.TARGETID}/S{sect}TIC{lc.TARGETID}_LC.png')
    pg = lc.normalize(unit='ppm').to_periodogram(minimum_period = 0.042, oversample_factor=300)
    period = pg.period_at_max_power
    pg1 = lc.normalize(unit='ppm').to_periodogram(maximum_period = 2.1*period.value, oversample_factor=100)
    mini = .7*period
    maxi = 1.3*period
    midpt = (mini + maxi)/2
    b = pg1.plot(view='period')
    b.figure.savefig(f'LightCurves/S{sect}TIC{lc.TARGETID}/S{sect}TIC{lc.TARGETID}_Periodogram.png')
    midpt = redef(mini, maxi, midpt, lc)
    folded = lc.fold(midpt)
    c = folded.scatter(label=fr'Period = {midpt.value:.5f} d')
    plt.close('all')
    c.figure.savefig(f'LightCurves/S{sect}TIC{lc.TARGETID}/S{sect}TIC{lc.TARGETID}_Folded.png')
    images = [Image.open(x) for x in [f'./LightCurves/S{sect}TIC{lc.TARGETID}/S{sect}TIC{lc.TARGETID}_LC.png', f'./LightCurves/S{sect}TIC{lc.TARGETID}/S{sect}TIC{lc.TARGETID}_Periodogram.png', f'./LightCurves/S{sect}TIC{lc.TARGETID}/S{sect}TIC{lc.TARGETID}_Folded.png']]
    widths, heights = zip(*(i.size for i in images))
    total_height = sum(heights)
    min_width = min(widths)
    new_im = Image.new('RGB', (min_width, total_height))
    y_offset = 0
    for im in images:
        new_im.paste(im, (0,y_offset))
        y_offset += im.size[1]
    new_im.save(f'./LightCurves/S{sect}TIC{lc.TARGETID}/S{sect}TIC{lc.TARGETID}.png')

#Saves the images as well as prints it out
def foldsaveprint(lcurve, sect):
    '''
    Folds,outputs, and saves the original light curve, periodogram, folded light curve, and all 3 combined
    Into a folder in the LightCurves folder
    
        Parameters:
            lcurve (lightkurve.LightCurve): a lightkurve.LightCurve object
            sect (int): the sector that the light curve is in
    '''
    lightc = lcurve
    a = lightc.scatter()
    lc = lightc[lightc.quality==0]
    a.figure.savefig(f'LightCurves/S{sect}TIC{lc.TARGETID}/S{sect}TIC{lc.TARGETID}_LC.png')
    pg = lc.normalize(unit='ppm').to_periodogram(minimum_period = 0.042, oversample_factor=300)
    period = pg.period_at_max_power
    pg1 = lc.normalize(unit='ppm').to_periodogram(maximum_period = 2.1*period.value, oversample_factor=100)
    mini = .7*period
    maxi = 1.3*period
    midpt = (mini + maxi)/2
    b = pg1.plot(view='period')
    b.figure.savefig(f'LightCurves/S{sect}TIC{lc.TARGETID}/S{sect}TIC{lc.TARGETID}_Periodogram.png')
    midpt = redef(mini, maxi, midpt, lc)
    folded = lc.fold(midpt)
    c = folded.scatter(label=fr'Period = {midpt.value:.5f} d')
    c.figure.savefig(f'LightCurves/S{sect}TIC{lc.TARGETID}/S{sect}TIC{lc.TARGETID}_Folded.png')
    images = [Image.open(x) for x in [f'./LightCurves/S{sect}TIC{lc.TARGETID}/S{sect}TIC{lc.TARGETID}_LC.png', f'./LightCurves/S{sect}TIC{lc.TARGETID}/S{sect}TIC{lc.TARGETID}_Periodogram.png', f'./LightCurves/S{sect}TIC{lc.TARGETID}/S{sect}TIC{lc.TARGETID}_Folded.png']]
    widths, heights = zip(*(i.size for i in images))
    total_height = sum(heights)
    min_width = min(widths)
    new_im = Image.new('RGB', (min_width, total_height))
    y_offset = 0
    for im in images:
        new_im.paste(im, (0,y_offset))
        y_offset += im.size[1]
    new_im.save(f'./LightCurves/S{sect}TIC{lc.TARGETID}/S{sect}TIC{lc.TARGETID}.png')


# Combines the seperate 3 pngs into one png
def combinefiles(ticid,sect):
    '''
    Combines the 3 individual graphs (original, periodogram, and folded light curve) of the light curve into one png
    
        Parameters:
            ticid (int): the ticid of the light curve
            sect (int): the sector that the light curve is in
    '''
    images = [Image.open(x) for x in [f'./LightCurves/S{sect}TIC{lc.TARGETID}/S{sect}TIC{lc.TARGETID}_LC.png', f'./LightCurves/S{sect}TIC{lc.TARGETID}/S{sect}TIC{lc.TARGETID}_Periodogram.png', f'./LightCurves/S{sect}TIC{lc.TARGETID}/S{sect}TIC{lc.TARGETID}_Folded.png']]
    widths, heights = zip(*(i.size for i in images))

    total_height = sum(heights)
    max_width = max(widths)
    min_width = min(widths)

    new_im = Image.new('RGB', (min_width, total_height))

    y_offset = 0
    for im in images:
        new_im.paste(im, (0,y_offset))
        y_offset += im.size[1]

    new_im.save(f'./LightCurves/S{sect}TIC{lc.TARGETID}/S{sect}TIC{lc.TARGETID}.png')


# Function to output original graph, periodogram, and phasefolded graph given TIC ID and their sector
def graphfoldprint(lcurve, sect):
    '''
    Folds and outputs the original light curve, periodogram, and the folded light curve
    Into a folder in the LightCurves folder
    
        Parameters:
            lcurve (lightkurve.LightCurve): a lightkurve.LightCurve object
            sect (int): the sector that the light curve is in
    '''
    lightc = lcurve
    a = lightc.scatter()
    lc = lightc[lightc.quality==0]
    pg = lc.normalize(unit='ppm').to_periodogram(minimum_period = 0.042, oversample_factor=300)
    period = pg.period_at_max_power
    pg1 = lc.normalize(unit='ppm').to_periodogram(maximum_period = 2.1*period.value, oversample_factor=100)
    mini = .7*period
    maxi = 1.3*period
    midpt = (mini + maxi)/2
    b = pg1.plot(view='period')
    midpt = redef(mini, maxi, midpt, lc)
    folded = lc.fold(midpt)
    c = folded.scatter(label=fr'Period = {midpt.value:.5f} d')

# Function to save the original graph, periodogram, and phasefolded graph in the 
# LightCurves folder without combining them
def graphfoldsave(lcurve, sect):
    '''
    Folds, prints, and saves the original light curve, periodogram, and folded light curve
    Into a folder in the LightCurves folder without combining them
    
        Parameters:
            lcurve (lightkurve.LightCurve): a lightkurve.LightCurve object
            sect (int): the sector that the light curve is in
    '''
        
    lightc = lcurve
    a = lightc.scatter()
    lc = lightc[lightc.quality==0]
    a.figure.savefig(f'LightCurves/S{sect}TIC{lc.TARGETID}/S{sect}TIC{lc.TARGETID}_LC.png')
    pg = lc.normalize(unit='ppm').to_periodogram(minimum_period = 0.042, oversample_factor=300)
    period = pg.period_at_max_power
    pg1 = lc.normalize(unit='ppm').to_periodogram(maximum_period = 2.1*period.value, oversample_factor=100)
    mini = .7*period
    maxi = 1.3*period
    midpt = (mini + maxi)/2
    b = pg1.plot(view='period')
    b.figure.savefig(f'LightCurves/S{sect}TIC{lc.TARGETID}/S{sect}TIC{lc.TARGETID}_Periodogram.png')
    midpt = redef(mini, maxi, midpt, lc)
    folded = lc.fold(midpt)
    c = folded.scatter(label=fr'Period = {midpt.value:.5f} d')
    c.figure.savefig(f'LightCurves/S{sect}TIC{lc.TARGETID}/S{sect}TIC{lc.TARGETID}_Folded.png')


# cleanlightcurve - light curve without noise
# phasecurve - phasefolded light curve
# smoothcurve - median filtered version of phasecurve
# cleanlcmod - median filtered version of cleanlightcurve


#adjusts min max and midpoint
def redef(mini, maxi, midpt, lc):
    '''
    Adjusts the midpt after comparing the standard deviation of the residuals to find the best period
    
        Parameters:
            mini (double): current minimum period for binary search
            maxi (double): current maximum period for binary search
            midpt (double): current assumed period value
            lc (lightkurve.LightCurve): a lightkurve.LightCurve object
        
        Returns: 
            midpt (double): the best period to phase fold on
    '''
    #tests if a multiple of the midpt is better than the current one
    twomidptresstddev = calcresidualstddevmidpt(lc, 2*midpt)
    midptresstddev = calcresidualstddevmidpt(lc, midpt)
    
    if (twomidptresstddev) < (midptresstddev):
        midpt = 2*midpt
        mini = .7*midpt
        maxi = 1.3*midpt
    
    #global mini, maxi, midpt, lc
    while(mini.value + 0.0001 < maxi.value):
        minresstddev = calcresidualstddevmin(lc, mini)
        midptresstddev = calcresidualstddevmidpt(lc, midpt)
        maxresstddev = calcresidualstddevmax(lc, maxi)
        if (minresstddev - midptresstddev) < (maxresstddev - midptresstddev):
            maxi = midpt
            midpt = (mini + maxi)/2
        else:
            mini = midpt
            midpt = (mini + maxi)/2
    
    return midpt

#calculates the std dev of residuals of period given by argument (num)
def calcresidualstddevmin(lc, num):
    lcfolded = lc.fold(num)
    cleanlightcurve = lcfolded[lcfolded.quality==0]
    phasecurve = lc.fold(num)[:]
    cleanlcmod = cleanlightcurve[:]
    cleanlcmod.flux = scipy.signal.medfilt(cleanlightcurve.flux, kernel_size=13)
    residual = cleanlcmod.flux - cleanlightcurve.flux
    ressqr = 0
    for x in range(len(cleanlcmod)):
        ressqr = ressqr + (residual[x] ** 2)
    minresstddev = math.sqrt((ressqr)/(len(cleanlcmod)-2))
    return minresstddev

def calcresidualstddevmax(lc, num):
    lcfolded = lc.fold(num)
    cleanlightcurve = lcfolded[lcfolded.quality==0]
    phasecurve = lc.fold(num)[:]
    cleanlcmod = cleanlightcurve[:]
    cleanlcmod.flux = scipy.signal.medfilt(cleanlightcurve.flux, kernel_size=13)
    residual = cleanlcmod.flux - cleanlightcurve.flux
    ressqr = 0
    for x in range(len(cleanlcmod)):
        ressqr = ressqr + (residual[x] ** 2)
    maxresstddev = math.sqrt((ressqr)/(len(cleanlcmod)-2))
    return (maxresstddev)

def calcresidualstddevmidpt(lc, num):
    lcfolded = lc.fold(num)
    cleanlightcurve = lcfolded[lcfolded.quality==0]
    phasecurve = lc.fold(num)[:]
    cleanlcmod = cleanlightcurve[:]
    cleanlcmod.flux = scipy.signal.medfilt(cleanlightcurve.flux, kernel_size=13)
    residual = cleanlcmod.flux - cleanlightcurve.flux
    ressqr = 0
    for x in range(len(cleanlcmod)):
        ressqr = ressqr + (residual[x] ** 2)
    midptresstddev = math.sqrt((ressqr)/(len(cleanlcmod)-2))
    return (midptresstddev)
