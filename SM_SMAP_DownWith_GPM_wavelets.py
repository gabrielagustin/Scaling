# -*- coding: utf-8 -*-
#!/usr/bin/python

"""
Created on Tue May 30 11:58:04 2017
@author: gag 

Script that applies the wavelet transform to modify the spatial resolution of satellite images. 
It uses two image sources, one with low resolution and another with better resolution.

"""

from osgeo import gdal, ogr, gdalconst
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
import numpy as np
import functions
import scipy as sci
from sklearn.metrics import mean_squared_error
import W2D
import scipy.signal as sg
from scipy import ndimage
import filters

matplotlib.use('GTKAgg')

dir = "...."


fechaSMAP = []
fechaGPM = []

fechaSMAP.append("2015-04-11")
fechaGPM.append("2015-04-08")

fechaSMAP.append("2015-05-02")
fechaGPM.append("2015-05-02")

fechaSMAP.append("2015-05-10")
fechaGPM.append("2015-05-12")

fechaSMAP.append("2015-05-26")
fechaGPM.append("2015-05-26")

fechaSMAP.append("2015-06-03")
fechaGPM.append("2015-06-05")

fechaSMAP.append("2015-06-19")
fechaGPM.append("2015-06-19")

fechaSMAP.append("2015-07-24")
fechaGPM.append("2015-07-23")

fechaSMAP.append("2015-08-17")
fechaGPM.append("2015-08-16")

fechaSMAP.append("2015-08-30")
fechaGPM.append("2015-08-30")

fechaSMAP.append("2015-09-10")
fechaGPM.append("2015-09-09")

fechaSMAP.append("2015-09-23")
fechaGPM.append("2015-09-23")

fechaSMAP.append("2015-10-28")
fechaGPM.append("2015-10-27")

fechaSMAP.append("2015-11-13")
fechaGPM.append("2015-11-10")

fechaSMAP.append("2015-11-21")
fechaGPM.append("2015-11-20")

fechaSMAP.append("2015-12-18")
fechaGPM.append("2015-12-14")

fechaSMAP.append("2015-12-28")
fechaGPM.append("2015-12-28")

fechaSMAP.append("2016-01-08")
fechaGPM.append("2016-01-07")

fechaSMAP.append("2016-01-19")
fechaGPM.append("2016-01-21")

fechaSMAP.append("2016-01-27")
fechaGPM.append("2016-01-31")

fechaSMAP.append("2016-02-14")
fechaGPM.append("2016-02-14")

fechaSMAP.append("2016-02-25")
fechaGPM.append("2016-02-24")

fechaSMAP.append("2016-03-12")
fechaGPM.append("2016-03-09")

fechaSMAP.append("2016-03-20")
fechaGPM.append("2016-03-19")

fechaSMAP.append("2016-04-02")
fechaGPM.append("2016-04-02")

fechaSMAP.append("2016-04-13")
fechaGPM.append("2016-04-12")

fechaSMAP.append("2016-04-24")
fechaGPM.append("2016-04-26")

fechaSMAP.append("2016-05-20")
fechaGPM.append("2016-05-20")


for ii in range(0, len(fechaSMAP)):

    ## se abre la imagen SMAP_SM
    #path1 = "/media/"+dir+"/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/SMAP/SMAP-Cali_val/"+fechaSMAP[ii]+"/recorte/"
    #nameFile1 = "SM.img"

    path1 = "/media/gag/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/SMAP/SMAP-Cali_val/"+fechaSMAP[ii]+"/subset_reprojected.data/"
    nameFile1 = "soil_moisture.img"

    src_ds_SMAP_SM, SMAP_SM, GeoTSMAP_SM, ProjectSMAP_SM = functions.openFileHDF(path1, nameFile1, 1)
    #
    #print "geo:"+ str(GeoTSMAP_SM)
    #print "pro: " + str(ProjectSMAP_SM)
    #
    #
    #print "Tamanio SMAP_SM"
    #print SMAP_SM.shape


    ## se abre la imagen de PP acumulada 7 dias de GPM a 11 Km
    path2 = "/media/"+dir+"/TOURO Mobile/GPM/"+fechaGPM[ii]+"/"
    nameFile2 = "recorte.img"
    ## se abre la imagen
    src_ds_PP, PP, GeoTPP, ProjectPP = functions.openFileHDF(path2, nameFile2, 1)
    #
    #print "geo:"+ str(GeoTPP)
    #print "pro: " + str(ProjectPP)

    PP = PP * 0.01
    PP = (PP-np.mean(PP)) /(np.max(PP)-np.min(PP))
    print "Tamanio PP 11km"
    print PP.shape

    #### se descompone la imagen PP 2 niveles
    #typeW = "dau"
    #typeW = "coif"
    #typeW = "syms"
    typeW = "haar"
    orden = 2
    if (typeW == "dau"):
        low, high = filters.filterDaubechies(orden)
    if (typeW == "coif"):
        low, high = filters.filterCoiflets(orden)
    if (typeW == "syms"):
        low, high = filters.filterSymlets(orden)
    if (typeW == "haar"):
        low, high = filters.filterHaar()

    level = 2
    lowi = functions.invFilter(low)
    highi = functions.invFilter(high)


    desc_PP40Km = W2D.dW2D(PP, highi, lowi, level)

    #
    #fig = plt.figure(1)
    #fig1 = fig.add_subplot(111)
    #fig1.set_title('Descompiscion ' +str(level)+ ' Niveles')
    #fig1.imshow(desc_PP40Km, cmap=cm.gray,interpolation='none')
    #

    #### se separa la aproximacion
    nRow, nCol = desc_PP40Km.shape
    nn = 2**(level)
    aprox = np.zeros((nRow/nn, nCol/nn))
    for i in range(0,nRow/nn):
        for j in range(0,nCol/nn):
            aprox[i, j] = desc_PP40Km[i, j]


    #fig = plt.figure(2)
    #fig2 = fig.add_subplot(111)
    #fig2.set_title('PP Aprox 40 Km')
    #fig2.imshow(aprox, cmap=cm.gray,interpolation='none')
    #
    #print "tamanio aprox VOD 40 Km"
    #print aprox.shape

    ##### se convierte la aproximacion NDVI en formato raster
    nRow, nCol = aprox.shape
    ds = gdal.GetDriverByName('MEM').Create('', nCol, nRow, 1, gdal.GDT_Float64)
    ds.SetProjection(ProjectPP)
    GeoT  = (GeoTPP[0], (2**level)*GeoTPP[1], GeoTPP[2], GeoTPP[3], GeoTPP[4], (2**level)*GeoTPP[5])
    #print nuevo
    ds.SetGeoTransform(GeoT)
    ds.GetRasterBand(1).WriteArray(np.array(aprox))
    PP_40km = ds.ReadAsArray()

    typeI = "Nearest"
    #typeI = "Bilinear"

    ##### se realiza la operacion de match entre ambas fuentes de datos
    ##### la imagen SMAP_VOD a 5km se machea con la imagen MODIS NDVI a 5Km
    nRow, nCol = PP_40km.shape
    data_src = src_ds_SMAP_SM
    data_match = ds
    match = functions.matchData(data_src, data_match, typeI, nRow, nCol)
    #match = functions.matchData(data_src, data_match, "Bilinear")
    #match = functions.matchData(data_src, data_match, "Cubic")
    band_match = match.ReadAsArray()

    #
    #fig = plt.figure(3)
    #fig3 = fig.add_subplot(111)
    #fig3.set_title('match')
    #fig3.imshow(band_match , cmap=cm.gray,interpolation='none')
    #


    ##### se crea la imagen ensambre que posee lo siguiente
    #####  Modis 60  det1 L8
    #####  det2 L8   det 3 L8
    r,c = band_match.shape

    ensamble = np.copy(desc_PP40Km)



    for i in range(0, r):
        for j in range(0, c):
            ensamble[i,j] = band_match[i, j]

    ensamble2 = np.copy(ensamble)

    #fig = plt.figure(9)
    #fig9 = fig.add_subplot(111)
    #fig9.set_title('VOD a 250m')
    #fig9.imshow (ensamble, cmap=cm.gray,interpolation='none')
    #


    imgRec = W2D.rW2D(ensamble2, high, low, level)


    #fig = plt.figure(10)
    #fig10 = fig.add_subplot(111)
    #fig10.set_title('VOD a 250m')
    #fig10.imshow (imgRec, cmap=cm.gray,interpolation='none')
    #
    ##### se guarda la imagen SM reescalada a 250m a partir del VOD a 250m
    nRow, nCol = imgRec.shape
    driver = gdal.GetDriverByName('GTiff')
    new_ds = driver.Create( "/media/"+dir+"/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/SMAP/SMAP-Cali_val/"+fechaSMAP[ii]+"/recorte/SM10K_W"+typeW+str(orden)+".img" ,nCol, nRow, 1, gdal.GDT_Float64)
    new_ds.SetGeoTransform(GeoTPP)
    new_ds.SetProjection(ProjectPP)
    new_ds.GetRasterBand(1).WriteArray(np.array(imgRec))

    #fig = plt.figure(0)
    #fig0 = fig.add_subplot(111)
    #fig0.set_title('SM SMAP')
    #fig0.imshow(SMAP_SM, cmap=cm.gray,interpolation='none')
    #


    #plt.show()


