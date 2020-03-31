# asl-denoising
**Combined Denoising and Suppression of Transient Artefacts in Arterial Spin Labelling MRI Using Deep Learning**

## Table of contents
* [General info](#general-info)
* [Software](#software)
* [Usage](#usage)

## General info
This repository contains the Python code needed to create the denoising autoencoder (DAE) model described in "*Combined Denoising and Suppression of Transient Artefacts in Arterial Spin Labelling MRI Using Deep Learning*" by Hales *et al.* Note this paper is currently under review and the model is subject to change. Full details are given in the *manuscript_github.pdf* file.

Current model architecture:
![image](https://user-images.githubusercontent.com/24695126/77834561-882fd000-713d-11ea-8ada-b4eef7958751.png)

**Figure 1**. Architecture of the denoising auto-encoder model, with an example low-SNR, single repetition dMraw image (left), and the corresponding high-SNR dMmean image (right; same axial slice averaged over 10 repetitions). A total of 30,960 image pairs were used to train and validate the model. Image dimensions are shown for each step, along with the number of filter layers used. Skip connections are illustrated as horizontal lines, convolution operations (with subsequent ReLU activation) as green arrows, and max-pooling / up-sampling operations as red / purple arrows respectively. 

## Software
This project was created with:
* Python 3.7.4
* Keras 2.3.1

The full list of requirements are given in requirements.txt, as part of this package

## Usage
Example usage of the DAE model class is as follows (using default settings):
```
from asl_denoising import Dae
dae = Dae()       # creates the DAE object, containing the default raw dataset, and info about the trained DAE model
dae.applyModel()  # apply the DAE model to the raw data
dae.showSlice()   # show an example raw/denoised imaging slice side-by-side
dae.saveResults() # saves the denoised dataset
```

Details of the DAE model class are given below (these can be accessed using help(Dae)):

    Attributes
    ----------
    dMRawFile: str
        The nifti file containing the noisy dM 3D dataset (default = 'dMnoisy.nii.gz', included in this package).
    modelFile: str
        The h5 model file to apply to noisy data (default = 'ADmodel_skipconnections_N_30960_BS_100_EP_100_stackNorm.h5',
        included in this package)
    modelMetaData: str
        The pickle file containing the metadata for the model - generally the same as the modelFile, but with '.pkl'
        extension (default = 'ADmodel_skipconnections_N_30960_BS_100_EP_100_stackNorm.pkl')

    Methods
    -------
    applyModel()
        Applies the denoising model (defined on class initialisation) to the raw data
    saveResults(saveFile=None)
        Saves the denoised dataset to a new nifti file, specified by saveFile (default: saveFile=<dMRawFile>_dae.nii.gz)
    showSlice(slice=None)
        Shows an example slice (specified by slice arg) of the raw and denoised datasets side by side (slice default = zres/2)








