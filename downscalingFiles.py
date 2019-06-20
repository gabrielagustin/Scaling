# -*- coding: utf-8 -*-
#!/usr/bin/python

"""
Created on Sun Jun 04 19:41:04 2017

@author: gag 

Script that allows apply the operation of Downscaling to a set of images, indicating:
 - n steps of downscaling
 - type and order of the filter to be used

Type filter: "db": Daubechies, "Symlets", "Haar", "Coiflets", "CDF", "Morlet"

"""

import os
import downscaling
import numpy as np


fechaModis = []

fechaModis.append("2015-06-26")
fechaModis.append("2015-07-28")
fechaModis.append("2015-10-16")
fechaModis.append("2015-12-19")
fechaModis.append("2016-03-21")


#ficheros = os.listdir(path)
##print len(ficheros)
#for i in range(0, len(ficheros)):

for i in range(0, len(fechaModis)):
    #print i
    #print "Archivo: " + str(ficheros[i])
    path = "/.../Modis/" + fechaModis[i] + "/"
    nameFile = "NDVI_reproyectado_recortado"

    # parametros nTimes, typeFilter, orderFilter, path, nameFileIn, orderFilter
    # Type filter:  "db": Daubechies, "Symlets", "Haar", "Coiflets", "CDF", "Morlet"

    #downscaling.applyDownscalingN(3,"Coiflets", 4, path, nameFile)
    #downscaling.applyDownscalingN(3,"Symlets", 4, path, nameFile)
    #downscaling.applyDownscalingN(3,"db", 4, path, nameFile)
    downscaling.applyDownscalingN(3,"Haar", 4, path, nameFile)









