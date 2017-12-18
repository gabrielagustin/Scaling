# -*- coding: utf-8 -*-
#!/usr/bin/python

import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as plt
import pywt
from numpy import fft
import functions


# Ondita Haar

def filterHaar():
    print pywt.wavelist('haar')
    wavelet = pywt.Wavelet('haar')
    return np.array(wavelet.dec_lo), np.array(wavelet.dec_hi)
    #return low.flatten(), high.flatten()


# Ondita daubechies

def filterDaubechies(orden):
    print pywt.wavelist('db')
    oo = "db"+str(orden)
    wavelet = pywt.Wavelet(oo)
    low = functions.invFilter(wavelet.dec_lo)
    high = functions.invFilter(wavelet.dec_hi)
    return low, high

# Ondita Symlets

def filterSymlets(orden):
    print pywt.families()
    print pywt.wavelist('sym')
    oo = "sym"+str(orden)
    wavelet = pywt.Wavelet(oo)
    low = functions.invFilter(wavelet.dec_lo)
    high = functions.invFilter(wavelet.dec_hi)
    return low, high

# Ondita Coiflets

def filterCoiflets(orden):
    print pywt.families()
    print pywt.wavelist('coif')
    oo = "coif"+str(orden)
    wavelet = pywt.Wavelet(oo)
    low = functions.invFilter(wavelet.dec_lo)
    high = functions.invFilter(wavelet.dec_hi)
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



def filterBior():
    print pywt.families()
    #print pywt.wavelist('bior')
    #wavelet = pywt.Wavelet('bior1.1')
    #wavelet = pywt.Wavelet('bior6.8')
    print pywt.wavelist('rbio')
    wavelet = pywt.Wavelet('rbio1.1')
    #wavelet = pywt.Wavelet('rbio6.8')
    return np.array(wavelet.dec_lo), np.array(wavelet.dec_hi)









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




def invFilter (filt):
    N = len(filt)
    invFilt = np.zeros(N)
    for i in range(0,N):
        invFilt[i] = filt[N-i-1]
    return invFilt



if __name__ == "__main__":
    #type = "Haar"
    #low, high = filterHaar()

    type = "Daubechies"
    low, high = filterDaubechies(2)

    #type = "Symlets"
    #low, high = filterSymlets(2)

    #type = "Coiflets"
    #low, high = filterCoiflets(4)

    #type = "Biortogonal"
    #low, high = filterBior()

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