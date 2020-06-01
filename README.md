# asl-denoising
**Combined Denoising and Suppression of Transient Artefacts in Arterial Spin Labelling MRI Using Deep Learning**

## Table of contents
* [General info](#general-info)
* [Package Contents](#package-contents)
* [Required Models](#required-modules)
* [Usage](#usage)

## General info
This repository contains the Python code needed to create the denoising autoencoder (DAE) model described in "*Combined Denoising and Suppression of Transient Artefacts in Arterial Spin Labelling MRI Using Deep Learning*" by Hales *et al., JMRI (in-press)* 

Model architecture:
![image](https://user-images.githubusercontent.com/24695126/77834561-882fd000-713d-11ea-8ada-b4eef7958751.png)

**Figure 1**. Architecture of the denoising auto-encoder model, with an example low-SNR, single repetition dMraw image (left), and the corresponding high-SNR dMmean image (right; same axial slice averaged over 10 repetitions).  Image dimensions are shown for each step, along with the number of filter layers used. Skip connections are illustrated as horizontal lines, convolution operations (with subsequent ReLU activation) as green arrows, and max-pooling / up-sampling operations as red / purple arrows respectively. 

## Package Contents
* AslDenoising.py
  * code needed to implement the DAE on raw ASL difference images (referred to as dM images - can be 2D, 3D or 4D)
* DAE.py:                       
  * code for creating the DAE model in Keras
* DaeTrainedModel.h5            
  * the trained DAE model (this is implemented when using AslDenoising.py)
* DaeTrainedModelMetaData.pkl   
  * the meta-data for the trained model 
* dMRaw.nii.gz                  
  * example nifti file containing raw dM images

## Required Modules
* numpy
* matplotlib
* nibabel
* keras
* skimage

## Usage
Details of the DAE model class are given below (these can be accessed using help(Dae)):

    Class used to implement the trained Denoising Autoencoder model on new data

    * Data can be passed as either a numpy array (using the dMRaw argument) or a NIFTI file (using the dMRawFile argument).
        - If both are specified, dMRawFile will override dMRaw
        - If neither are specified, the example dataset included in this package will be loaded (dMRaw.nii.gz)

    Attributes
    ----------
    dMRaw: numpy array
        A numpy array containing the raw ASL difference images (can be 2D, 3D, or 4D)

    dMRawFile: str
        A nifti file containing the raw ASL difference images (default = 'dMRaw.nii.gz', included in this package).

    Methods
    -------
    applyModel()
        Applies the denoising model to the raw data

    showSlice(slice=None, rep=None)
        Shows an example slice of the raw and denoised datasets side by side
            - specify the slice number to display using the slice argument (default = central slice)
            - specify the repetition (for 4D data only) using the rep argument (default = first repetition)

    showTraining()
        Shows a plot of the training and validation loss during model training

Example usage of the DAE model class is as follows :
```
    from AslDenoising import Dae
    dae = Dae()                                 # creates an instance of the Dae object, using the default raw dM dataset
    dae = Dae(dMRaw=<numpy array>)              # creates an instance of the Dae object, using a numpy array as containing the raw dataset
    dae = Dae(dMRawFile=<path to NIFTI file>)   # creates an instance of the Dae object, using a nifti file to load the raw dataset
    dae.applyModel()                            # apply the DAE model to the raw dataset
    dae.showSlice(slice=10)                     # display the raw and denoised dM images side-by-side, at slice position 10
    dae.showTraining()                          # display the training and validation loss during model training
```










