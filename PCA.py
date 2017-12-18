import numpy as np
import functions
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import cm


path = "/media/gag/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/Landsat8/L_2015-06-18/"
nameFile = "NDVI_recortado"

src_ds_L8, bandL8, GeoTL8, Project = functions.openFileHDF2(path, nameFile, 1)
print bandL8.shape

path = "/media/gag/TOURO Mobile/Trabajo_Sentinel_NDVI_CONAE/MODIS/2015-06-26/"
nameFile = "NDVI_reprojectado_recortado"
src_ds_Modis, bandModis, GeoTModis, Project = functions.openFileHDF2(path, nameFile, 1)
print bandModis.shape

#fig = plt.figure(10)
#fig1 = fig.add_subplot(111)
#fig1.imshow(bandModis[:512,:512], cmap=cm.gray)


### a la imagen modis se le cambia la resolucion mediante el match
### se puede interpolar segun los metodos Nearest, Bilinear o Cubic

data_src = src_ds_Modis
data_match = src_ds_L8
match = functions.matchData(data_src, data_match, "Bilinear")
#match = functions.matchData(data_src, data_match, "Nearest")
#match = functions.matchData(data_src, data_match, "Cubic")
band_match = match.ReadAsArray()
#modis = band_match

#bandL8 = bandL8[:2048,:2048]
#modis = band_match[:2048,:2048]


bandL8 = bandL8[:512,:512]
modis = band_match[:512,:512]

X = bandL8
mu = np.mean(X, axis=0)

nComp = 200

pca = PCA(nComp)
pca.fit(X)
#print pca.explained_variance_ratio_
#print (pca.explained_variance_ratio_).shape
#print np.sum(pca.explained_variance_ratio_)

### se busca el nro de componente para el que la variabilidad explicada sea igual al 60%
sum = 0
idx = 0
a = np.array(pca.explained_variance_ratio_)
for i in range (0, nComp):
    if (sum < 0.60):
        sum = sum +  a[i]
    else:
        idx = i
        break
print "Porcentaje de varianza alcanzado: " + str(sum)
print "Cantidad de componentes: "+ str(idx)


#### pca.transform: Apply dimensionality reduction to X
#### pca.components: Principal axes in feature space, representing the
#### directions of maximum variance in the data. The components are sorted by explained_variance_.

for i in range(1,idx):
    #i = 1
    imgRec = np.dot(pca.transform(X)[:,:i], pca.components_[:i,:])
    imgRec += mu
imgRec = (imgRec + modis)/2.0
    #Xhat = pca.components_[:i,:]
    #print Xhat.shape

fig = plt.figure(0)
fig1 = fig.add_subplot(111)
fig1.imshow(bandL8, cmap=cm.gray)

fig = plt.figure(11)
fig11 = fig.add_subplot(111)
fig11.set_title('hist band L8 ')
bandDownNew = bandL8.flatten('C')
y, x, _  = plt.hist(bandDownNew, 256, facecolor='b')

fig = plt.figure(12)
fig11 = fig.add_subplot(111)
fig11.set_title('hist imgRec ')
bandDownNew = imgRec.flatten('C')
y, x, _  = plt.hist(bandDownNew, 256, facecolor='b')

fig = plt.figure(1)
fig1 = fig.add_subplot(111)
fig1.imshow(modis, cmap=cm.gray)

fig = plt.figure(3)
fig1 = fig.add_subplot(111)
fig1.imshow(imgRec, cmap=cm.gray)

fig = plt.figure(4)
fig1 = fig.add_subplot(111)
fig1.set_title('Error')
fig1.imshow (np.abs(bandL8-imgRec), cmap=cm.gray)


error = functions.imgError(bandL8,imgRec)

print "RMSE:"+str(error)


L8 = bandL8.flatten('C')
mm = imgRec.flatten('C')

v1 = L8.tolist()
v2 = mm.tolist()
z = np.polyfit(v1,v2, 1)
g = np.poly1d(z)
cor = np.corrcoef(v1,v2)[0,1]
if (cor > 0):
    cor=(cor)*(cor)
else:
    cor=(cor*(-1))*(cor*(-1))

fig = plt.figure(7)
fig7 = fig.add_subplot(111)
fig7.scatter(L8, imgRec, s=5)
x = np.linspace(*fig7.get_xlim())
fig7.plot(x, x, 'k--')

fig7.text(-0.5, 1, 'r^2=%5.3f' % cor, fontsize=15)
fig7.plot(v1,g(v1),'r--')




#fig1.imshow(band_match, cmap=cm.gray)


plt.show()
