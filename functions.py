# -*- coding: utf-8 -*-
#!/usr/bin/python

import numpy as np
import scipy.signal as sg
from osgeo import gdal, ogr, gdalconst
import sys
import matplotlib.pyplot as plt
import scipy.ndimage as nimg
import pywt
from numpy import fft
from scipy.ndimage import convolve


# funcion que realiza el submuestreo de las columnas en una matriz

def subMuestreoColumnas(matriz):
    # señal sobremuestreadax
    # tamaño de las filas
    lRows, lColumns = matriz.shape
    sSig = np.zeros((lRows, lColumns / 2))
    #print sSig.shape
    f = 0
    for j in range(0, lColumns):
        if (j % 2 == 1):
            sSig[:,f] = matriz[:, j]
            f = f + 1
    return sSig


# funcion que realiza el submuestreo de las filas en una matriz
def subMuestreoFilas(matriz):
    # señal sobremuestreadax
    # tamaño de las filas
    lRows, lColumns = matriz.shape
    sSig = np.zeros((lRows/2, lColumns))
    #print sSig.shape
    f = 0
    for i in range(0, lRows):
        if (i % 2 == 1):
            sSig[f, :] = matriz[i, :]
            f = f + 1
    return sSig




# funcion que realiza el sobremuestreo de las columnas en una matriz, mediante
# el agregado de ceros

def sobreMuestreoColumnas(matriz):
    # señal sobremuestreadax
    # tamaño de las filas
    lRows, lColumns = matriz.shape
    sSig = np.zeros((lRows, lColumns * 2))
    #print sSig.shape
    for j in range(0, lColumns * 2 , 2):
        sSig[:, j] = matriz[:, (j / 2) - 1]
    return sSig



# funcion que realiza el sobremuestreo de las filas en una matriz, mediante
# el agregado de ceros

def sobreMuestreoFilas(matriz):
    # señal sobremuestreada
    # tamaño de las filas
    lRow, lColumns = matriz.shape
    sSig = np.zeros((lRow * 2, lColumns))
   # se crea una fila de ceros
    for i in range(0, lRow * 2 , 2):
        sSig[i, :] = matriz[(i / 2) - 1, :]
    #print sSig
    return sSig




# esta funcion aplica los filtros pasados como parametros a la señal
# recibida utilizando la convolucion circular con retardo, compensando de
# esta manera el retardo introducido por el filtro

def applyFilterRet(x,h, desp):
    retardo = int(np.floor(len(h)/2.0))
    #print retardo
    xh = convCircularRet(x, h, retardo, desp)
    return xh

def convCircularRet(x,h,r, desp):
    N =len(x)
    M=len(h)
    #y = np.zeros(N)
    ## se realiza con convolucion
    #for k in range(0,N):
        #y[k] = 0;
        #for l in range(0,M):
            #indice= (np.mod((N+k-l+r), N))
            #y[k]=y[k]+(h[l]*x[indice])
    #x = rotate(x,r)
    #hpad = np.concatenate((h, np.zeros(N-M)), axis =0)
    #y = (np.fft.ifft( np.fft.fft(x)*np.fft.fft(hpad)).real)
    y = convolve(x, h, mode='wrap')
    return y


def rotate(l, r):
   if len(l) == 0:
      return l
   #y = -y % len(l)     # flip rotation direction
   return np.concatenate((l[r:], l[:r]))

#def convCircularRet(x,h,r, desp):
    #desp = 1
    #b = h
    ##print p
    ##if not(np.mod(p,2) == 0):
    #xx = np.zeros(2 * r + len(x))
    ### copy all the elements
    #lenx = len(x)
    #for i in range(0,lenx):
        #xx[i + r] = x[i]
    #for i in range(0,r):
        #xx[i] = x[r - i]
    #lenxx = len(xx)
    ##### Se agrega al inicio y al fin de la señal los valores en forma de espejo
    #for i in range(1,r +1):
        #xx[lenx + r + i -1 ] = x[lenx-1 - i]
    ##print xx
    ##y = np.zeros(len(x))
    ##for i in range(0,len(y)):
        ##y[i] =xx[i + r + desp]
    ##print y
    #yy = sg.convolve(xx,b, "same")
    #y = np.zeros(len(x))
    #for i in range(0,len(y)):
        #y[i] = yy[i + r + desp]
    #return y

#x = [1,2,3,4,5,6,7,8,9]
#print x
#h = np.ones(5)
##h = h * 3
##print h
#r = 3
#convCircularRet(x,h,r,1)



def openFileHDF(path, nameFileIn, nroBand):
    #print "Open File"
    file = str(path + nameFileIn)
    #print file
    try:
        src_ds = gdal.Open(file)
    except RuntimeError, e:
        print 'Unable to open File'
        print e
        sys.exit(1)

    cols = src_ds.RasterXSize
    rows = src_ds.RasterYSize
    #print cols
    #print rows
    bands = src_ds.RasterCount
    #print bands

    # se obtienen las caracteristicas de las imagen HDR
    GeoT = src_ds.GetGeoTransform()
    #print GeoT
    Project = src_ds.GetProjection()

    try:
        srcband = src_ds.GetRasterBand(nroBand)
    except RuntimeError, e:
        # for example, try GetRasterBand(10)
        print 'Band ( %i ) not found' % band_num
        print e
        sys.exit(1)
    band = srcband.ReadAsArray()
    nRow, nCol = band.shape
    nMin = np.min((nRow,nCol))
    #print "minimo: "+ str(nMin)
    nMin = int(nMin/2)*2
    #print "minimo par: "+ str(nMin)
    factor = int(np.log(nMin)/np.log(2))
    #print "factor: " + str(factor)
    tamanio = 2**factor
    #print "tamanio:" + str(tamanio)
    band = band[:tamanio,:tamanio]
    #### creo src_ds con nuevo tamanio
    #src_ds = gdal.GetDriverByName('MEM').Create('', tamanio, tamanio, 1, gdal.GDT_Float64)
    #src_ds.SetProjection(Project)
    #geotransform = GeoT
    #src_ds.SetGeoTransform(geotransform)
    #src_ds.GetRasterBand(1).WriteArray(np.array(band))
    #band = src_ds.ReadAsArray()
    return src_ds, band, GeoT, Project

def openFileHDF2(path, nameFileIn, nroBand):
    #print "Open File"
    file = str(path + nameFileIn)
    #print file
    try:
        src_ds = gdal.Open(file)
    except RuntimeError, e:
        print 'Unable to open File'
        print e
        sys.exit(1)

    cols = src_ds.RasterXSize
    rows = src_ds.RasterYSize
    #print cols
    #print rows
    bands = src_ds.RasterCount
    #print bands

    # se obtienen las caracteristicas de las imagen HDR
    GeoT = src_ds.GetGeoTransform()
    #print GeoT
    Project = src_ds.GetProjection()

    try:
        srcband = src_ds.GetRasterBand(nroBand)
    except RuntimeError, e:
        # for example, try GetRasterBand(10)
        print 'Band ( %i ) not found' % band_num
        print e
        sys.exit(1)
    band = srcband.ReadAsArray()
    return src_ds, band, GeoT, Project




def matchData(data_src, data_match, type, nRow, nCol):
    # funcion que retorna la informacion presente en el raster data_scr
    # modificada con los datos de proyeccion y transformacion del raster data_match
    # se crea un raster en memoria que va a ser el resultado
    #data_result = gdal.GetDriverByName('MEM').Create('', data_match.RasterXSize, data_match.RasterYSize, 1, gdalconst.GDT_Float64)

    data_result = gdal.GetDriverByName('MEM').Create('', nCol, nRow, 1, gdalconst.GDT_Float64)

    # Se establece el tipo de proyección y transfomcion en resultado  qye va ser coincidente con data_match
    data_result.SetGeoTransform(data_match.GetGeoTransform())
    data_result.SetProjection(data_match.GetProjection())

    # se cambia la proyeccion de data_src, con los datos de data_match y se guarda en data_result
    if (type == "Nearest"):
        gdal.ReprojectImage(data_src,data_result,data_src.GetProjection(),data_match.GetProjection(), gdalconst.GRA_NearestNeighbour)
    if (type == "Bilinear"):
        gdal.ReprojectImage(data_src, data_result, data_src.GetProjection(), data_match.GetProjection(), gdalconst.GRA_Bilinear)
    if (type == "Cubic"):
        gdal.ReprojectImage(data_src, data_result, data_src.GetProjection(), data_match.GetProjection(), gdalconst.GRA_Cubic)

    return data_result


# funcion que crea un archivo HDF basado en los datos Geotransform y Projection
# de la imagen original, recibe ademas el nombre del archivo de salida, el tipo
# de archivo a crear, la imagen y su taman

def createHDFfile(path, nameFileOut, driver, img, xsize, ysize, GeoT, Projection):
    print "archivo creado:" + str(nameFileOut)
    driver = gdal.GetDriverByName(driver)
    ds = driver.Create(path + nameFileOut, xsize, ysize, 1, gdal.GDT_Float64)
    ds.SetProjection(Projection)
    geotransform = GeoT
    ds.SetGeoTransform(geotransform)
    ds.GetRasterBand(1).WriteArray(np.array(img))
    return

def latlonMatrix(GeoT, band):
    ### retorna las matrices de latitud y longitud
    print "Crea matriz de Latitud y Longitud"
    rows, cols = band.shape
    lat = np.zeros((rows, cols))
    for i in range(0,rows):
        for j in range(0,cols):
            lat[i,j] = float(GeoT[0])+ j*float(GeoT[1])
    lon = np.zeros((rows, cols))
    for i in range(0,rows):
        for j in range(0,cols):
            lon[i,j] = float(GeoT[3])+ i*float(GeoT[5])
    return lat, lon



def corregistration(GeoTb1, band1, GeoTb2, band2, punto):
    ####
    print "Corregistracion"
    rowsb1, colsb1 = band1.shape
    latb1 = np.zeros((rowsb1, colsb1))
    lonb1 = np.zeros((rowsb1, colsb1))
    latb1, lonb1 = latlonMatrix(GeoTb1, band1)

    print latb1

    rowsb2, colsb2 = band2.shape
    latb2 = np.zeros((rowsb2, colsb2))
    lonb2 = np.zeros((rowsb2, colsb2))
    latb2, lonb2 = latlonMatrix(GeoTb2, band2)

    print latb2

    difRows = np.abs(rowsb1 -rowsb2)
    difCols = np.abs(colsb1 - colsb2)

    print "Diferencia en Filas: " + str(difRows)
    print "Diferencia en columnas: " + str(difCols)

    lon_b1 = lonb1[punto,punto]
    #print lon_b1
    lat_b1 = latb1[punto,punto]
    #print lat_b1
    error = 0
    for i in range(0, rowsb2):
        for j in range(0, colsb2):
            r1 = np.abs(lonb2[i,j]- lon_b1)
            #print r1
            r2 = np.abs(latb2[i,j]- lat_b1)
            #print r2
            errorNew = r1 + r2
            #print errorNew
            if ((i == 0) and (j == 0)):
                error = errorNew
                imin = i
                jmin = j
            if (errorNew < error):
                #print "SI"
                error = errorNew
                imin = i
                jmin = j
    deltai = np.abs(imin-punto)
    deltaj = np.abs(jmin-punto)
    return deltai, deltaj, error



def invFilter (filt):
    N = len(filt)
    invFilt = np.zeros(N)
    for i in range(0,N):
        invFilt[i] = filt[N-i-1]
    return invFilt

def imgError(original,imgRec):
    error = np.sqrt(np.sum((original-imgRec)**2))/(original.shape[0]*original.shape[1])
    return error



if __name__ == "__main__":
    #type = "Haar"
    #low, high = filterHaar(4)

    #type = "Daubechies"
    #low, high = filterDaubechies(8)

    #type = "Symlets"
    #low, high = filterSymlets(8)

    #type = "Coiflets"
    #low, high = filterCoiflets(4)

    type = "Biortogonal"
    low, high = filterBior()

    #type = "Morlet"
    #low, high = filterMorlet(8)


    #low, high = filterCDF(8)

    #points = 100
    #a = 4
    #low = sg.bspline(points, a)
    #high = sg.qmf(low)

    fig = plt.figure(1)
    fig.suptitle(type)
    fig1 = fig.add_subplot(211)
    fig1.set_title('low')
    fig1.plot(low)
    fig2 = fig.add_subplot(212)
    fig2.set_title('high')
    fig2.plot(high)
    #hi = invFilter(high)
    #li = invFilter(low)
    #fig = plt.figure(2)
    #fig1 = fig.add_subplot(211)
    #fig1.set_title('low')
    #fig1.plot(li)
    #fig2 = fig.add_subplot(212)
    #fig2.set_title('high')
    #fig2.plot(hi)

    plt.show()
