# Scaling

An algorithm to perform the rescaling of the satellite images is presented. The techniques used are: wavelet transforms (WT) and principal component analysis (PCA).


* Fusion using WT, the image with greater detail is first decomposed, the approximation coefficients are replaced by the adaptation of the less detailed image and finally the reconstruction process is applied. The adaptation of the minor detail image is done by resampling using the nearest neighbor method. The scheme used is presented in the following image:

<p align="center">
  <img width=650 src="fusionWavelets.png"/>
</p>

* Fusion using PCA, the adaptation of the low resolution image to the hi resolution image is performed, that is, pixels size is modified by an interpolation method; in this case, the nearest neighbor method was used. Then the decompositions of both images are combined, exchanging the first coefficient of the image with fine spatial resolution for the first component of the image with coarse spatial resolution, as detailed in the scheme presented in the following image. Finally, the image reconstruction is performed.

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
