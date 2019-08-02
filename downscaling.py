# -*- coding: utf-8 -*-
#!/usr/bin/python

"""
Created on Tue May 30 11:58:04 2017
@author: gag 

"""



import matplotlib.pyplot as plt
import functions
import numpy as np
from osgeo import gdal, ogr
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
import scipy.misc

import W2D


def downscaling(aprox, det1, det2, det3, highF, lowF, desp):
    """ Function that performs 1 level of reconstruction
    Parameters:
    -----------
    aprox, det1, det2, det3 : full path of the raster image
    highF, highL:
    nroBand: number of the band to read
    Returns:
    --------
    src_ds: raster
    band: raster as arrays
    GeoT: raster georeference info
    Project: projections
    """

    imRec = []
    sizeF = len(lowF)
    #print sizeF

    #las matrices de entrada se sobremuestrean por columnas
    aprox2C = functions.sobreMuestreoColumnas(aprox)
    det12C = functions.sobreMuestreoColumnas(det1)
    det22C = functions.sobreMuestreoColumnas(det2)
    det32C = functions.sobreMuestreoColumnas(det3)

    lrow, lcol = aprox2C.shape
    aproxL = aprox2C
    det1H = det12C
    det2L = det22C
    det3H = det32C

    for i in range(0, lrow):
        # a cada fila de aprox se le aplica ef filtro low
        aproxL[i, :] = functions.applyFilterRet(aprox2C[i, :], lowF, desp)
        det1H[i, :] = functions.applyFilterRet(det12C[i, :], highF, desp)
        det2L[i, :] = functions.applyFilterRet(det22C[i, :], lowF, desp)
        det3H[i, :] = functions.applyFilterRet(det32C[i, :], highF, desp)


    # se realiza la suma de las matrices luego de aplicar los filtros
    det3Hdet2L = det3H+det2L
    det1HAproxL = det1H+aproxL

    #aprox2L = functions.repeatF(aprox2L,n)
    # a estas dos matrices se las sobremuestrea por filas
    R1 = functions.sobreMuestreoFilas(det3Hdet2L)
    R2 = functions.sobreMuestreoFilas(det1HAproxL)

    R1L = R1
    R2H = R2
    # se le aplica el filtro por columnas y se crean
    # dos matrices nuevas
    lrow, lcolumn = R1.shape
    for j in range(0, lcolumn):
        # se le aplica el filtro h
        R1L[:, j] = functions.applyFilterRet(R1[:, j].T, highF , desp)
        R2H[:, j] = functions.applyFilterRet(R2[:, j].T, lowF, desp)
    # con las matrices obtenidas se calcula la imagen reconstruida
    imRec = (R1L + R2H)
    imRec = imRec * 0.5

    return imRec

###############################################################################


def applyDownscalingN(nTimes, typeFilter, orderFilter, path, nameFileIn):
    """
    Función que realiza la operación de downscaling nVeces
    Recibe: numero de veces, tipo de filtro, orden del filtro, directorio y nombre
    del archivo HDF
    Retorna: imagen HDF
    """
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
    aprox = nBand * 255.0
    maxI = np.max(aprox)
    minI = np.min(aprox)
    print "maximo: " + str(maxI)
    print "minimo: " + str(minI)
    for i in range(0, nTimes):
        ###
        #np.random.seed(i)
        print "paso n:" + str( i + 1)
        nRow, nCol = aprox.shape
        det1 = det2 = det3 = np.zeros((nRow, nCol))
        det1 = det2 = det3 = (np.random.rand(nRow, nCol))*2.0-1.0
        #print det1
        #det1 = ((np.random.rand(nRow, nCol))*2-1)#*255
        ##print det1
        #det2 = ((np.random.rand(nRow, nCol))*2-1)#*255
        ##print det2
        #det3 = ((np.random.rand(nRow, nCol))*2-1)#*255
        ##print det3


        #### acá debo modificar
        ### rW2D

        img = downscaling(aprox, det1, det2, det3, high, low, desp)
        imgNew = np.array(img)

        nRow, nCol = imgNew.shape
        aprox = imgNew
        aprox = aprox / 255.0
    text = "_" + "Down_2_" + "_" + str(nCol) + '_' + str(nRow)

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

    nameFile = "lena.jpg"
    #path = "/media/ggarcia/TOURO Mobile/Scaling/img/"
    #nameFile = "img_georeference_subset"
    path = "/media/gag/TOURO Mobile/Scaling/img/"

    #nameFile = "NDVI_reprojectado_recortado"
    #path = "/media/gag/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/MODIS/2014-12-19/"
    """
    Types of filters available
    """
    applyDownscalingN(1,"Coiflets", 4, path, nameFile)

