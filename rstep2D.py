# -*- coding: utf-8 -*-
#!/usr/bin/python
"""
Created on Tue Jul 12 20:16:04 2016


@author: gag

Esta funcion realiza un nivel de reconstruccion 2D
recibe la matrices de descomposicion aprox, det1, det2, det3, los filtros
y el mapa de colores

"""



import matplotlib.pyplot as plt
import functions
import numpy as np
from osgeo import gdal, ogr
import matplotlib.pyplot as plt
from matplotlib import cm
import sys

import scipy.misc





def rstep2D(aprox,det1,det2,det3,h,g):
    imRec=[]

    #las matrices de entrada se sobremuestrean por columnas
    aprox = functions.sobreMuestreoColumnas(aprox)
    det1 = functions.sobreMuestreoColumnas(det1)
    det2 = functions.sobreMuestreoColumnas(det2)
    det3 = functions.sobreMuestreoColumnas(det3)

    lrow, lcol = aprox.shape

    aproxH = aprox
    det1G = det1    g = [1 - np.sqrt(3), - 3 + np.sqrt(3), 3 + np.sqrt(3), - 1 - np.sqrt(3)]
    g = np.array(g)
    det2H = det2
    det3G = det3

    for i in range(0, lrow):
        # a cada fila se le aplican los filtros g y h
        det3G[i, :] = functions.applyFilterRet(det3[i, :], g)
        det2H[i, :] = functions.applyFilterRet(det2[i, :], h)
        det1G[i, :] = functions.applyFilterRet(det1[i, :], g)
        aproxH[i, :] = functions.applyFilterRet(aprox[i, :], h)

    # se realiza la suma de las matrices luego de aplicar los filtros
    d3Gd2H = det3G + det2H
    d1GApH = det1G + aproxH

    # a estas dos matrices se las sobremuestrea por filas
    d3Gd2H = functions.sobreMuestreoFilas(d3Gd2H)
    d1GApH = functions.sobreMuestreoFilas(d1GApH)

    # a estas matrices se le aplican los filtros por columnas y se crean
    # dos matrices nuevas
    d3Gd2HG = d3Gd2H
    d1GApHH = d1GApH
    lrow, lcolumn = d1GApH.shape
    for j in range(0, lcolumn):
        # a la matriz que es  la suma de det3G y det2H sobremuestreada(d3dG2H)
        # se le aplica el filtro g
        d3Gd2HG[:, j] = functions.applyFilterRet(d3Gd2H[:, j].T, g)
        # a la matriz que es la suma de det1G y aproxH sobremuestreada (d1GApH)
        # se le aplica el filtro h
        d1GApHH[:, j] = functions.applyFilterRet(d1GApH[:, j].T,h)

    # con las matrices obtenidas se calcula la imagen reconstruida
    #imRec=4*(d3Gd2HG+d1GApHH);
    imRec=(d3Gd2HG+d1GApHH)

   # si graph es igual a 1 se grafica la imagen reconstruida
    #if(graph == 1)
        #figure(22)
        #image(imRec);
        #map=gray(128);
        #colormap(map)
    return imRec

###############################################################################
# inicio main


[h,g]=functions.onditaDaubechies(1)

low, high = functions.filterDaubechies(8)


# this allows GDAL to throw Python Exceptions
gdal.UseExceptions()

path = "/home/gag/Desktop/Downscaling/img/"


# "NDVI_reprojectado_recortado"
# "lena.jpg"
#Tvis-animated.gif
try:
    src_ds = gdal.Open(path + "NDVI_reprojectado_recortado")
except RuntimeError, e:
    print 'Unable to open File'
    print e
    sys.exit(1)

cols = src_ds.RasterXSize
rows = src_ds.RasterYSize
bands = src_ds.RasterCount


try:
    srcband = src_ds.GetRasterBand(1)
except RuntimeError, e:
    # for example, try GetRasterBand(10)
    print 'Band ( %i ) not found' % band_num
    print e
    sys.exit(1)


lc = srcband.ReadAsArray()



nRow, nCol = lc.shape
print "Tamaño original:" + str(nRow)+ " - " + str(nCol)
aprox = lc
det1 = det2 = det3 = np.zeros((nRow, nCol))

numSteps = 2
for i in range(0,numSteps):
    ###
    print "paso n:" + str(i+1)
    img = rstep2D(aprox, det1, det2, det3, low, high)
    img = zip(*img[::-1])
    img = zip(*img[::-1])

    imgNew = np.array(img)

    nRow, nCol = imgNew.shape
    print nRow, nCol
    aprox = imgNew
    det1 = det2 = det3 = np.zeros((nRow, nCol))



scipy.misc.imsave(path+str('ndviNew.jpg'), img)
#plt.imshow (img, cmap=cm.gray)


plt.show()
