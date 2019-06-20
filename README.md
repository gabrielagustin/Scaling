# Scaling

An algorithm to perform the rescaling of the satellite images is presented. The techniques used are: wavelet transforms (WT) and principal component analysis (PCA).


* In order to carry out the fusion using WT, the image with greater detail is first decomposed, the approximation coefficients are replaced by the adaptation of the less detailed image and finally the reconstruction process is applied. The adaptation of the minor detail image is done by resampling using the nearest neighbor method. The scheme used is presented in the following image:

<p align="center">
  <img width=650 src="fusionWavelets.png"/>
</p>

* 

<p align="center">
  <img width=650 src="fusionPCA.png"/>
</p>


Dependences:
    
    python - Gdal
    python - pywt
    python - sklearn
    python - scipy
    python - NumPy
    python - Matplolib


Pages sources:

  PyWavelets: https://pywavelets.readthedocs.io/en/latest/

  PCA decomposition: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

  More details: https://ieeexplore.ieee.org/document/7996007
