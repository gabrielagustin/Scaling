# -*- coding: utf-8 -*-
#!/usr/bin/python

import matplotlib.pyplot as plt
import functions
import numpy as np
from osgeo import gdal, ogr
import matplotlib.pyplot as plt
from matplotlib import cm
import sys

import scipy.misc


def downscaling(aprox, highF, lowF, desp):
    imRec = []
    sizeF = len(lowF)
    #print sizeF

    #las matrices de entrada se sobremuestrean por columnas
    aprox2 = functions.sobreMuestreoColumnas(aprox)

    lrow, lcol = aprox2.shape
    aprox2L = aprox2

    for i in range(0, lrow):
        # a cada fila de aprox se le aplica ef filtro low
        aprox2L[i, :] = functions.applyFilterRet(aprox2[i, :], lowF, desp)

    #aprox2L = functions.repeatF(aprox2L,n)
    # a estas dos matrices se las sobremuestrea por filas
    aprox2L2 = functions.sobreMuestreoFilas(aprox2L)
    aprox2L2L = aprox2L2

    # se le aplica el filtro por columnas y se crean
    # dos matrices nuevas
    lrow, lcolumn = aprox2L2L.shape
    for j in range(0, lcolumn):
        # se le aplica el filtro h
        aprox2L2L[:, j] = functions.applyFilterRet(aprox2L2[:, j].T, lowF, desp)

    # con las matrices obtenidas se calcula la imagen reconstruida
    imRec = (aprox2L2L)
    imRec = imRec * 2.0

    return imRec

###############################################################################


def applyDownscalingN(nTimes, typeFilter, orderFilter, path, nameFileIn):

    if (typeFilter == "db"):
        low, high = functions.filterDaubechies(orderFilter)
        desp = 0
    if (typeFilter == "Symlets"):
        low, high = functions.filterSymlets(orderFilter)
        low = low.T
        high = high.T
        if (orderFilter == 2):
            desp = 1
        if (orderFilter == 4):
            desp = 3
    if (typeFilter == "Coiflets"):
        low, high = functions.filterCoiflets(orderFilter)
        low = low.T
        high = high.T
        if (orderFilter == 2):
            desp = 0
        if (orderFilter == 4):
            desp = - 1

    if (typeFilter == "Haar"):
        low, high = functions.filterHaar(orderFilter)
        desp = 0
        low = low.T
        high = high.T

    if (typeFilter == "Morlet"):
        low, high = functions.filterMorlet(orderFilter)
        desp = 2
        low = low.T
        high = high.T
    #print len(high)
    #print high
    if (typeFilter == "CDF"):
        low, high = functions.filterCDF(orderFilter)
        desp = 2
        low = low.T
        high = high.T

    # this allows GDAL to throw Python Exceptions
    gdal.UseExceptions()

    src_ds, band, GeoT, Project = functions.openFileHDF(path, nameFileIn, 1)

    print "Downscaling"
    nRow, nCol = band.shape
    nBand = np.array(band)
    #nRow, nCol = nBand.shape
    print nRow, nCol
    print "Tamaño original: " + str(nRow)+ " - " + str(nCol)
    aprox = nBand
    maxI = np.max(aprox)
    minI = np.min(aprox)
    print "maximo: " + str(maxI)
    print "minimo: " + str(minI)
    for i in range(0, nTimes):
        ###
        print "paso n:" + str( i + 1)
        img = downscaling(aprox, high, low, desp)
        imgNew = np.array(img)

        nRow, nCol = imgNew.shape
        aprox = imgNew
    text = "_" + "Down" + "_" +"_" + str(nCol) + '_' + str(nRow)

    # se crea un nuevo archivo HDR, con el mismo header pero diferente banda
    nameFileOut = str(nameFileIn + text + "_" + str(typeFilter) + "_" +str(orderFilter))
    # + str("_band_2")
    nuevo = (GeoT[0], GeoT[1]/(2**float(nTimes)), GeoT[2], GeoT[3], GeoT[4], GeoT[5]/(2**float(nTimes)))
    #print nuevo
    GeoT = nuevo

    print "Tamaño Final: " + str(nRow)+ " - " + str(nCol)
    functions.createHDFfile(path, nameFileOut, 'ENVI', aprox, nCol, nRow, GeoT, Project)
    src_ds = None


if __name__ == "__main__":

    #nameFile = "lena.jpg"
    #path = "/media/ggarcia/TOURO Mobile/Scaling/img/"
    #nameFile = "img_georeference_subset"
    #path = "/media/gag/TOURO Mobile/Scaling/img/"

    #nameFile = "NDVI_reprojectado_recortado"
    #path = "/media/gag/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/MODIS/2014-12-19/"
    #path = "/media/gag/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/MODIS/2015-06-26/"
    #nameFile = "NDVI_reprojectado_recortado"

    #path = "/media/ggarcia/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/MODIS/2015-06-26/"
    #nameFile = "NDVI_reprojectado_recortado"

    path = "/media/gag/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/MODIS/2015-06-10/"
    nameFile = "NDVI_reproyectado_recortado"
    #path = "/media/ggarcia/TOURO Mobile/MOD02QKM/2015-06-18/"
    #nameFile = "NDVI_recortado"

    applyDownscalingN(3,"db", 2, path, nameFile)