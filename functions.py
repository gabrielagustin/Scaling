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




#m = np.ones((10,10))
##print m
#print m.shape
#print m
#newM = subMuestreoColumnas(m)
#print newM
#print newM.shape
#newM = subMuestreoFilas(newM)
#print newM
#print newM.shape




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


#m = np.ones((10,10))
##print m
#print m.shape
#newM = sobreMuestreoColumnas(m)
#print newM
#print newM.shape


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


#m = np.ones((10,10))
#print m
#print m.shape
#newM = sobreMuestreoFilas(m)
#print newM
#print newM.shape




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





# la funcion devuelve los filtros que si se normalizan dan la ondita de
# Daubechies de orden 4
# se normalizan cuando se recibe normalize igual a 1

def onditaDaubechies(normalize):
    h = []
    g = []
    # filtro h
    h = [1 + np.sqrt(3), 3 + np.sqrt(3), 3 - np.sqrt(3), 1 - np.sqrt(3)]
    h = np.array(h)
    h = (1 / 8.0) * h
    # filtro  g
    g = [1 - np.sqrt(3), - 3 + np.sqrt(3), 3 + np.sqrt(3), - 1 - np.sqrt(3)]
    g = np.array(g)
    g = (1 / 8.0) * g
    if (normalize == 1):
        h = h / np.linalg.norm(h)
        g = g / np.linalg.norm(g)
    return h, g


def filterDaubechies(orden):
    low = sg.daub(orden)
    # se convierte el filtro pasa bajos en pasa altos
    high = sg.qmf(low)
    high = high / np.linalg.norm(high)
    low = low / np.linalg.norm(low)
    return low, high



def filterHat(orden):
    a = 4
    low = sg.ricker(orden, a)
    high = sg.qmf(low)
    return low, high


def filterMorlet(orden):
    low = np.real(sg.morlet(orden, w = 5, s=1.0))
    # se convierte el filtro pasa bajos en pasa altos
    high = sg.qmf(low)
    return low, high



def filterDau():
    low = np.array([1+np.sqrt(3), 3+np.sqrt(3), 3-np.sqrt(3), 1-np.sqrt(3)])
    high = np.array([1-np.sqrt(3), -3+np.sqrt(3), 3+np.sqrt(3), -1-np.sqrt(3)])
    low = low * float(1/8.0)
    high = high * float(1/8.0)
    low = low/np.linalg.norm(low)
    high = high/np.linalg.norm(high)
    return low, high

def filterBior():
    print pywt.families()
    #print pywt.wavelist('bior')
    #wavelet = pywt.Wavelet('bior1.1')
    #wavelet = pywt.Wavelet('bior6.8')
    print pywt.wavelist('rbio')
    wavelet = pywt.Wavelet('rbio1.1')
    #wavelet = pywt.Wavelet('rbio6.8')
    return np.array(wavelet.dec_lo), np.array(wavelet.dec_hi)



# la funcion devuelve los filtros de promedio móvil y diferencia móvil
# normalizados dan lugar a la ondita Haar
# se normaliza si recibe normalize igual a 1
def filterHaar(orden):
    #print pywt.wavelist('coif')
    #wavelet = pywt.Wavelet('coif4')
    #return np.array(wavelet.dec_lo), np.array(wavelet.dec_hi)
    #print wavelet
    #low = np.zeros((2*orden,1))
    #high = np.zeros((2*orden,1))
    #for i in range(len(low)/2,len(low)):
        #low [i] = 0.5
        #high [i] = - 0.5
    #low = low - 0.5
    #high = high - 0.5
    low = np.zeros((2,1))
    high = np.zeros((2,1))
    low[0] = 0.7071067811865476
    low[1] = 0.7071067811865476
    high[0] = 0.7071067811865476
    high[1] = -0.7071067811865476

    return low.flatten(), high.flatten()



def filterSymlets(orden):
    if (orden == 2):
        low = np.zeros((4, 1))
        low[0] = 0.482962913
        low[1] = 0.836516304
        low[2] = 0.224143868
        low[3] = -0.129409523

        high = np.zeros((4, 1))
        high[0] = -0.129409523
        high[1] = -0.224143868
        high[2] = 0.836516304
        high[3] = -0.482962913

    if (orden == 4):
        low = np.zeros((8, 1))
        low[0] = 0.032223101
        low[1] = -0.012603967
        low[2] = -0.099219544
        low[3] = 0.297857796
        low[4] = 0.803738752
        low[5] = 0.497618668
        low[6] = -0.029635528
        low[7] = -0.075765715

        high = np.zeros((8, 1))
        high[0] = -0.075765715
        high[1] = 0.029635528
        high[2] = 0.497618668
        high[3] = -0.803738752
        high[4] = 0.297857796
        high[5] = 0.099219544
        high[6] = -0.012603967
        high[7] = -0.032223101
    return low.flatten(), high.flatten()

def filterCoiflets(orden):
    if (orden == 2):
        low = np.zeros((12, 1))
        low[0] = 0.016387336
        low[1] = -0.041464937
        low[2] = -0.067372555
        low[3] = 0.386110067
        low[4] = 0.812723635
        low[5] = 0.417005184
        low[6] = -0.076488599
        low[7] = -0.059434419
        low[8] = 0.023680172
        low[9] = 0.005611435
        low[10] = -0.001823209
        low[11] = -0.000720549

        high = np.zeros((12, 1))
        high[0] = -0.000720549
        high[1] = 0.001823209
        high[2] = 0.005611435
        high[3] = -0.023680172
        high[4] = -0.059434419
        high[5] = 0.076488599
        high[6] = 0.417005184
        high[7] = -0.812723635
        high[8] = 0.386110067
        high[9] = 0.067372555
        high[10] = -0.041464937
        high[11] = -0.016387336

    if (orden == 4):
        low = np.zeros((24, 1))
        low[0] = 0.000892314
        low[1] = -0.001629492
        low[2] = -0.007346166
        low[3] = 0.016068944
        low[4] = 0.0266823
        low[5] = -0.0812667
        low[6] = -0.056077313
        low[7] = 0.415308407
        low[8] = 0.782238931
        low[9] = 0.434386056
        low[10] = -0.066627474
        low[11] = -0.096220442
        low[12] = 0.039334427
        low[13] = 0.025082262
        low[14] = -0.015211732
        low[15] = -0.005658287
        low[16] = 3.75*10**-03
        low[17] = 1.27*10**-03
        low[18] = -0.000589021
        low[19] = -0.000259975
        low[20] = 6.23*10**-05
        low[21] = 3.12*10**-05
        low[22] = -3.26*10**-06
        low[23] = -1.78*10**-06

        high = np.zeros((24, 1))
        high[0] = -1.78*10**-06
        high[1] = 3.26*10**-06
        high[2] = 3.12*10**-05
        high[3] = -6.23*10**-05
        high[4] = -0.000259975
        high[5] = 0.000589021
        high[6] = 0.001266562
        high[7] = -0.003751436
        high[8] = -0.005658287
        high[9] = 0.015211732
        high[10] = 0.025082262
        high[11] = -0.039334427
        high[12] = -0.096220442
        high[13] = 0.066627474
        high[14] = 0.434386056
        high[15] = -0.782238931
        high[16] = 0.415308407
        high[17] = 0.056077313
        high[18] = -0.0812667
        high[19] = -0.0266823
        high[20] = 0.016068944
        high[21] = 0.007346166
        high[22] = -0.001629492
        high[23] = -0.000892314

    low = low/np.linalg.norm(low)
    high = high/np.linalg.norm(high)
    return low.flatten(), high.flatten()

### Cohen-Daubechies-Feauveau wavelet
def filterCDF(orden):
    low = np.zeros((9, 1))
    low[0] = 0
    low[1] = -0.091271763114
    low[2] = -0.057543526229
    low[3] = 0.591271763114
    low[4] = 1.11508705
    low[5] = 0.591271763114
    low[6] = -0.057543526229
    low[7] = -0.091271763114
    low[8] = 0

    high = np.zeros((9, 1))
    high[0] = 0.026748757411
    high[1] = 0.016864118443
    high[2] = -0.078223266529
    high[3] = -0.266864118443
    high[4] = 0.602949018236
    high[5] = -0.266864118443
    high[6] = -0.078223266529
    high[7] = 0.016864118443
    high[8] = 0.026748757411
    low = low/np.linalg.norm(low)
    high = high/np.linalg.norm(high)
    return low.flatten(), high.flatten()

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