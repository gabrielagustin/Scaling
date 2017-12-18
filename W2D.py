# -*- coding: utf-8 -*-
import functions
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import copy


def dstepW2D(img, highF, lowF):
    ### funcion que realiza la descomposicion Wavelet un nivel
    lRows, lColumns = img.shape
    #print img.shape
    imgH = np.zeros((lRows, lColumns))
    imgL = np.zeros((lRows, lColumns))
    for i in range(0, lRows):
        # a cada fila de aprox se le aplica ef filtro low and high
        imgL[i, :] = functions.applyFilterRet(img[i, :], lowF, 0)
        imgH[i, :] = functions.applyFilterRet(img[i, :], highF, 0)

    # a estas dos matrices se las submuestrea por columnas
    imgLSF = functions.subMuestreoColumnas(imgL)
    imgHSF = functions.subMuestreoColumnas(imgH)

    r,c = imgLSF.shape
    #print imgLSF.shape
    imgHSF_H = np.zeros((r, c))
    imgHSF_L = np.zeros((r, c))
    imgLSF_H = np.zeros((r, c))
    imgLSF_L = np.zeros((r, c))

    for j in range(0, c):
        # a cada columna  se le aplica ef filtro low and high
        imgHSF_H[:, j] = functions.applyFilterRet(imgHSF[:, j], highF, 0)
        imgHSF_L[:, j] = functions.applyFilterRet(imgHSF[:, j], lowF, 0)
        imgLSF_H[:, j] = functions.applyFilterRet(imgLSF[:, j], highF, 0)
        imgLSF_L[:, j] = functions.applyFilterRet(imgLSF[:, j], lowF, 0)


    # a estas dos matrices se las submuestrea por filas
    det3 = functions.subMuestreoFilas(imgHSF_H)*0.5
    det2 = functions.subMuestreoFilas(imgHSF_L)*0.5
    det1 = functions.subMuestreoFilas(imgLSF_H)*0.5
    aprox = functions.subMuestreoFilas(imgLSF_L)*0.5

    return aprox, det1, det2, det3



def dW2D(img, high, low, level):
    # funcion que realiza la descomposicion Wavelet N niveles
    aprox, det1, det2, det3 = dstepW2D(img, high, low)
    m1 = np.concatenate((aprox, det1),axis=0)
    m2 = np.concatenate((det2, det3),axis=0)
    mCoeff = np.concatenate((m1, m2), axis=1)
    for i in range(1,level):
        aprox, det1, det2, det3 = dstepW2D(aprox, high, low)
        m1 = np.concatenate((aprox, det1),axis=0)
        m2 = np.concatenate((det2, det3),axis=0)
        m = np.concatenate((m1, m2), axis=1)
        # esta nueva matriz se ubica en la posicion de la matriz aprox en mCoeff
        r, c = m.shape
        for i in range(0,r):
            for j in range(0,c):
                mCoeff[i,j] = m[i,j]
    return mCoeff

def rstepW2D(aprox, det1, det2, det3, highF, lowF):
    # las matrices de entrada se sobremuestrean por columnas
    ax = functions.sobreMuestreoColumnas(aprox)
    d1 = functions.sobreMuestreoColumnas(det1)
    d2 = functions.sobreMuestreoColumnas(det2)
    d3 = functions.sobreMuestreoColumnas(det3)

    r, c = ax.shape
    det3H = np.zeros((r, c))
    det2L = np.zeros((r, c))
    det1H = np.zeros((r, c))
    aproxL = np.zeros((r, c))

    for i in range (0,r):
        ## se aplica el filtro por filas
        det3H[i, :] = functions.applyFilterRet(d3[i, :], highF, 0)
        det2L[i, :] = functions.applyFilterRet(d2[i, :], lowF, 0)
        det1H[i, :] = functions.applyFilterRet(d1[i, :], highF, 0)
        aproxL[i, :] = functions.applyFilterRet(ax[i, :], lowF, 0)
    ## se suman las matrices luego de aplicar los filtros
    d3d2 = np.zeros((r,c))
    d3d2 = det3H + det2L
    d1aprox = np.zeros((r,c))
    d1aprox = det1H + aproxL
    ## a estas dos matrices se las sobremuestra por filas
    d3d2_n = functions.sobreMuestreoFilas(d3d2)
    d1aprox_n = functions.sobreMuestreoFilas(d1aprox)
    r, c = d3d2_n.shape
    d3d2H = np.zeros((r, c))
    d1aproxL = np.zeros((r, c))
    for j in range (0,c):
        #se aplican los filtros L y H por columnas
        d3d2H[:, j] = functions.applyFilterRet(d3d2_n[:, j], highF, 0)
        d1aproxL[:, j] = functions.applyFilterRet(d1aprox_n[:, j], lowF, 0)
    img = (d3d2H + d1aproxL)*2
    return img


def rW2D(matrix, highF, lowF, level):
    ### esta funcion realiza la reconstruccion Wavelet 2D a partir de la matriz
    ### con el siguiente formato
    ###   Aprox_L2    D1_L2       D1_L1
    ###     D2_L2     D3_L2
    ###           D2_L1          D3_L1
    r,c = matrix.shape
    #imRec = np.zeros((r,c))

    for k in range (level, 0, -1):
        print "nivel: " + str(k)
        ## se calculan los tamaños de las matrices de coeficientes del nivel
        ## inferior
        f = 2 **(k-1)
        #print f
        lRow = int(np.floor(r/(f)))
        lCol = int(np.floor(c/(f)))
        print lRow
        print lCol
        ## se obtienen las matrices aprox, det1, det2 y det3
        ap = np.zeros((lRow/2, lCol/2))
        #print ap.shape
        d1 = np.zeros((lRow/2, lCol/2))
        d2 = np.zeros((lRow/2, lCol/2))
        d3 = np.zeros((lRow/2, lCol/2))
        for i in range(0,lRow/2):
            for j in range(0,lCol/2):
                ap[i, j] = matrix[i, j]
            for j in range(lCol/2, lCol):
                d1[i, j - lCol/2] = matrix[i, j]
        for i in range(lRow/2, lRow):
            for j  in range(0, lCol/2):
                d2[i-lRow/2, j] = matrix[i, j]
            for j in range(lCol/2, lCol):
                d3[i-lRow/2, j- lCol/2] = matrix[i, j]
        ### se realiza la recontruccion a partir de las imagenes
        #fig1 = plt.figure(20*k)
        #fig1 = fig1.add_subplot(111)
        #fig1.imshow (d1, cmap=cm.gray,vmin=-0.1,vmax=0.2)
        #fig1.set_title('D1' + 'Nivel'+str(k))
        #fig2 = plt.figure(20*k+1)
        #fig2 = fig2.add_subplot(111)
        #fig2.imshow (d2, cmap=cm.gray,vmin=-0.1,vmax=0.2)
        #fig2.set_title('D2' + 'Nivel'+str(k))
        #fig3 = plt.figure(20*k+1)
        #fig3 = fig3.add_subplot(111)
        #fig3.imshow (d3, cmap=cm.gray,vmin=-0.1,vmax=0.2)
        #fig3.set_title('D3' + 'Nivel'+str(k))

        #fig = plt.figure(16)
        #bandDownNew = d1.flatten('C')
        #y, x, _  = plt.hist(bandDownNew, 256, facecolor='b')
        #plt.show()
        imRec = rstepW2D(ap, d1, d2, d3, highF, lowF)
        r2,c2 = imRec.shape
        #print imRec.shape
        for i in range(0,r2):
            for j in range(0,c2):
                matrix [i,j] = imRec[i,j]
    imRec = matrix
    return imRec


def returnAprox(matrix, level):
    return

if __name__ == "__main__":

    #nameFile = "ndvipru.jpg"
    #nameFile = "ndvireal"
    #nameFile = "lena.jpg"
    #path = "/media/ggarcia/TOURO Mobile/Scaling/img/"
    #src_ds, band, GeoT, Project = functions.openFileHDF(path, nameFile, 1)
    #path1 = "/media/ggarcia/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/Landsat8/L_2015-06-18/"
    #nameFile1 = "NDVI_recortado"
    ### se abre la imagen landsat8
    #src_ds, band, GeoTL8, ProjectL8 = functions.openFileHDF(path1, nameFile1, 1)

    path1 = "/media/ggarcia/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/Landsat8/L_2015-06-18/"
    nameFile1 = "NDVI_recortado"
    ## se abre la imagen landsat8
    src_ds, band, GeoT, Project = functions.openFileHDF(path1, nameFile1, 1)


    band = band[:600,:600]
    nRow, nCol = band.shape
    img = np.array(band)
    img = img/float(np.max(img))
    fig0 = plt.figure(1)
    fig0 = fig0.add_subplot(111)
    fig0.set_title('imagen original')
    fig0.imshow (img,cmap=cm.gray)

    fig1 = plt.figure(8)
    fig1 = fig1.add_subplot(111)
    fig1.set_title('hist original Img')
    bandDownNew = img.flatten('C')
    y, x, _  = plt.hist(bandDownNew, 256, facecolor='b')


    #imgNew = img.astype(np.uint8)
    r,c = img.shape
    imgNew = np.zeros((r,c))
    for i in range(0,r):
        for j in range(0,c):
            imgNew[i,j] = float(img[i,j])

    print "Tamaño original: " + str(nRow)+ " - " + str(nCol)
    low, high = functions.filterDaubechies(2)
    #low, high = functions.filterCoiflets(2)
    #low, high = functions.onditaDaubechies(1)
    lowi = functions.invFilter(low)
    highi = functions.invFilter(high)
    level = 3
    m = dW2D(imgNew, highi, lowi, level)
    #m = dW2D(img, high, low, level)
    #nRow, nCol = aprox.shape
    #print "Tamaño reducido: " + str(nRow)+ " - " + str(nCol)
    fig1 = plt.figure(2)
    fig1 = fig1.add_subplot(111)
    fig1.set_title('descomposicion 2D')
    fig1.imshow (m,cmap=cm.gray)


    m2 = copy.copy(m)
    #low, high = functions.filterDau()
    imgRec = rW2D(m2, high, low, level)
    print "Tamanio reconstruccion: " + str(imgRec.shape)
    fig2 = plt.figure(3)
    fig2 = fig2.add_subplot(111)
    fig2.set_title('reconstruccion 2D')
    fig2.imshow (imgRec, cmap=cm.gray)

    fig3 = plt.figure(9)
    fig3 = fig3.add_subplot(111)
    fig3.set_title('hist rec Img')
    bandDownNew = imgRec.flatten('C')
    y, x, _  = plt.hist(bandDownNew, 256, facecolor='b')
    #nRow, nCol = imgRec.shape
    #print "Tamaño recontruido: " + str(nRow)+ " - " + str(nCol)

    fig4 = plt.figure(4)
    fig4 = fig4.add_subplot(111)
    fig4.set_title('Error')
    fig4.imshow ((imgRec - img),cmap=cm.gray)

    fig5 = plt.figure(5)
    ax = fig5.add_subplot(111)
    #fig5.set_title('Error') imgRec[:-1,:]
    ax.scatter(img, imgRec, s=5)
    x = np.linspace(*ax.get_xlim())
    ax.plot(x, x, 'k--')

    print functions.imgError(img,imgRec)
    plt.show()