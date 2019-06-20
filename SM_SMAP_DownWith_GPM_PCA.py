# -*- coding: utf-8 -*-
#!/usr/bin/python

"""
Created on Tue May 30 11:58:04 2017
@author: gag 

Script that applies the PCA method to modify the spatial resolution of satellite images. 
It uses two image sources, one with low resolution and another with better resolution.

"""


import numpy as np
import functions
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import stats
from osgeo import gdal, ogr, gdalconst



#dir = "ggarcia"
dir = "gag"


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
#for ii in range(0,1):
    #ii = 0
    path = "/media/"+dir+"/TOURO Mobile/GPM/"+fechaGPM[ii]+"/"
    nameFile = "recorte.img"


    src_ds_GPM, bandGPM, GeoTGPM, ProjectGPM = functions.openFileHDF(path, nameFile, 1)
    #print bandGPM.shape

    bandGPM = bandGPM*0.1
    bandGPM = (bandGPM-np.mean(bandGPM)) /(np.max(bandGPM)-np.min(bandGPM))

    #fig = plt.figure(1)
    #fig1 = fig.add_subplot(111)
    #fig1.set_title('GPM PP')
    #fig1.imshow(bandGPM, cmap=cm.gray, interpolation='none')
    #
    #print "media GPM: "+ str(np.mean(bandGPM))
    #print "desvio GPM: "+ str(np.std(bandGPM))
    #print "var GPM: "+ str(np.var(bandGPM))




    #path = "/media/"+dir+"/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/SMAP/SMAP-Cali_val/"+fechaSMAP[ii]+"/recorte/"
    #nameFile = "SM.img"

    path = "/media/gag/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/SMAP/SMAP-Cali_val/"+fechaSMAP[ii]+"/subset_reprojected.data/"
    nameFile = "soil_moisture.img"
    src_ds_SM, bandSM, GeoTSM, ProjectSM = functions.openFileHDF(path, nameFile, 1)

    #print bandSM.shape

    #fig = plt.figure(3)
    #fig3 = fig.add_subplot(111)
    #fig3.set_title('SM SMAP')
    #fig3.imshow(bandSM, cmap=cm.gray, interpolation='none')
    #


    ### a la imagen SM de SMAP se le cambia la resolucion mediante el match
    ### se puede interpolar segun los metodos Nearest, Bilinear o Cubic

    nRow, nCol = bandGPM.shape

    type = "Nearest"
    #type = "Bilinear"
    data_src = src_ds_SM
    data_match = src_ds_GPM
    match = functions.matchData(data_src, data_match, type, nRow, nCol)
    band_match = match.ReadAsArray()
    #print band_match.shape
    ### band_match = bandSM

    #print "Tamanio band match: "+str(band_match.shape)

    #fig = plt.figure(10)
    #fig10 = fig.add_subplot(111)
    #fig10.set_title('SM SMAP matched with GPM')
    #fig10.imshow(band_match, cmap=cm.gray,interpolation='none')

    nComp = 5

    ### se aplica PCA a la imagen con resolucion fina
    pcaGPM = PCA(nComp)
    pcaGPM.fit(bandGPM)
    muGPM = np.mean(bandGPM, axis=0)
    #print muGPM
    GPMreduction = pcaGPM.transform(bandGPM)
    GPMcomponents = pcaGPM.components_
    print "variancia GPM" + str(pcaGPM.explained_variance_ratio_)



    ### se aplica PCA a la imagen con resolucion gruesa
    pcaSMAP = PCA(nComp)
    band_match[np.isnan(band_match)] = 0

    pcaSMAP.fit(band_match)
    muSMAP = np.mean(band_match, axis=0)
    #print muSMAP
    SMAPreduction = pcaSMAP.transform(band_match)
    SMAPcomponents = pcaSMAP.components_
    print "variancia SMAP" + str(pcaSMAP.explained_variance_ratio_)

    #imgRec = np.dot(SMAPreduction[:,:0], GPMcomponents[:0,:])
    ##for i in range(1,nComp):
    #imgRec += np.dot(SMAPreduction[:,1:nComp], SMAPcomponents[1:nComp,:])

    imgRec = np.zeros(band_match.shape)
    imgRec += np.dot(SMAPreduction[:,:1], GPMcomponents[:1,:])
    imgRec += np.dot(SMAPreduction[:,1:nComp], SMAPcomponents[1:nComp,:])
    imgRec += muSMAP

    fileName = "/media/"+dir+"/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/SMAP/SMAP-Cali_val/"+fechaSMAP[ii]+"/recorte/SM10Km_PCA_"+type+"_"+str(nComp)+".img"
    #### se guarda la imagen de SM reescalada a 10Km a partir de GPM
    nRow, nCol = imgRec.shape
    print imgRec.shape
    driver = gdal.GetDriverByName('GTiff')
    new_ds = driver.Create(fileName, nCol, nRow, 1, gdal.GDT_Float64)
    new_ds.SetGeoTransform(GeoTGPM)
    new_ds.SetProjection(ProjectGPM)
    new_ds.GetRasterBand(1).WriteArray(np.array(imgRec))


    #fig = plt.figure(13)
    #fig13 = fig.add_subplot(111)
    #fig13.set_title('SM down 10km')
    #fig13.imshow(imgRec, cmap=cm.gray, interpolation='none')



    plt.show()
#
