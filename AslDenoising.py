# Script for implementing a deep-learning-based denoising autoencoder (DAE) model on ASL datasets
# Version 1.0, created 29/05/2020
# Author: Patrick Hales

import numpy as np
from matplotlib import pyplot as plt
import nibabel as nib
import os
from keras.models import load_model
from skimage.transform import resize
import pickle

class Dae:
    """

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

    Examples
    --------
    from AslDenoising import Dae
    dae = Dae()                                 # creates an instance of the Dae object, using the default raw dM dataset
    dae = Dae(dMRaw=<numpy array>)              # creates an instance of the Dae object, using a numpy array as containing the raw dataset
    dae = Dae(dMRawFile=<path to NIFTI file>)   # creates an instance of the Dae object, using a nifti file to load the raw dataset
    dae.applyModel()                            # apply the DAE model to the raw dataset
    dae.showSlice(slice=10)                     # display the raw and denoised dM images side-by-side, at slice position 10
    dae.showTraining()                          # display the training and validation loss during model training

    """

    def __init__(self, dMRaw=None, dMRawFile=None):

        # define the files containing the trained model and the metadata for the model
        self.modelFile = 'DaeTrainedModel.h5'
        self.modelMetaData = 'DaeTrainedModelMetaData.pkl'

        # load the keras model
        self.model = load_model(self.modelFile)
        # load the model meta data
        f = open(self.modelMetaData, 'rb')
        self.mp = pickle.load(f)
        f.close()

        # get info about the training of the model
        self.loss = self.mp['loss']
        self.val_loss = self.mp['val_loss']
        self.epochs = np.arange(1, self.loss.size + 1)

        # load the raw data
        # If a path to the nifti file containing the raw data is not specified...
        if dMRawFile is None:
            if dMRaw is None:
                # load the default raw dM file, if no data array or nifti file is specified
                dMRawFile = 'dMRaw.nii.gz'
                nii = nib.load(dMRawFile)
                self.dMraw = nii.get_data().astype('float32')
            else:
                # load the data array stored in the dMRaw variable, passed by the user
                self.dMraw = dMRaw
        # If the path to the nifti file is specified...
        else:
            if dMRaw is not None:
                print('** Warning: both dMRaw and dMRawFile supplied. Using dMRawFile...')
            # load the raw nifti file supplied by the user
            nii = nib.load(dMRawFile)
            self.dMraw = nii.get_data().astype('float32')

        # get info about the structure of the raw data
        if self.dMraw.ndim == 4:
            [self.xres, self.yres, self.zres, self.tres] = self.dMraw.shape

        if self.dMraw.ndim == 3:
            [self.xres, self.yres, self.zres] = self.dMraw.shape
            self.tres = None

        if self.dMraw.ndim == 2:
            [self.xres, self.yres] = self.dMraw.shape
            self.zres = None
            self.tres = None

        if self.dMraw.ndim < 2 or self.dMraw.ndim > 4:
            print('** Error: input data must consist of a 2D, 3D or 4D array **')

    def applyModel(self):

        # the DAE model is expecting input images of size 128 x 128. If this is not the case, re-size the input
        self.rescale_res = 128
        self.xres0 = self.xres
        self.yres0 = self.yres

        if self.xres0 != self.rescale_res or self.yres0 != self.rescale_res:
            self.dMraw0 = self.dMraw
            self.xres = self.rescale_res
            self.yres = self.rescale_res

            if self.dMraw.ndim == 4:
                self.dMraw = np.zeros((self.rescale_res, self.rescale_res, self.zres, self.tres))
                for r in range(self.tres):
                    for k in range(self.zres):
                        thisSlice = self.dMraw0[:, :, k, r]
                        self.dMraw[:, :, k, r] = resize(thisSlice, (self.rescale_res, self.rescale_res), anti_aliasing=True)

            if self.dMraw.ndim == 3:
                self.dMraw = np.zeros((self.rescale_res, self.rescale_res, self.zres))
                for k in range(self.zres):
                    thisSlice = self.dMraw0[:, :, k]
                    self.dMraw[:, :, k] = resize(thisSlice, (self.rescale_res, self.rescale_res), anti_aliasing=True)

            if self.dMraw.ndim == 2:
                self.dMraw = resize(self.dMraw, (self.rescale_res, self.rescale_res), anti_aliasing=True)

        # Tidy the input data to remove outliers
        self.dMraw[np.isnan(self.dMraw)] = 0.0
        self.dMraw[np.isinf(self.dMraw)] = 0.0
        dat_test_vector = self.dMraw.flatten()
        uthr = np.percentile(dat_test_vector, 99.9)
        lthr = np.percentile(dat_test_vector, 0.1)
        self.dMraw = np.clip(self.dMraw, a_min=lthr, a_max=uthr)

        # re-order the input noisy data array, as Keras uses the format <image index, xres, yres, channels>
        if self.dMraw.ndim == 4:
            noisy_image_stack = np.zeros((self.zres * self.tres, self.xres, self.yres, 1))
            predicted_image_stack = np.zeros((self.zres * self.tres, self.xres, self.yres, 1))
            dMdenoised = np.zeros((self.xres, self.yres, self.zres, self.tres))
            # loop through all slices, and reformat the raw images as image stacks
            sctr = 0
            for t in range(self.tres):
                for k in range(self.zres):
                    a = self.dMraw[:, :, k, t]
                    noisy_image_stack[sctr, :, :, :] = a.reshape(self.xres, self.yres, 1)
                    sctr += 1

        if self.dMraw.ndim == 3:
            predicted_image_stack = np.zeros((self.zres, self.xres, self.yres, 1))
            noisy_image_stack = np.zeros((self.zres, self.xres, self.yres, 1))
            dMdenoised = np.zeros((self.xres, self.yres, self.zres))
            for k in range(self.zres):
                a = self.dMraw[:, :, k]
                noisy_image_stack[k, :, :, :] = (a.reshape(self.xres, self.yres, 1)).astype('float32')

        if self.dMraw.ndim == 2:
            noisy_image_stack = self.dMraw.reshape(1, self.xres, self.yres, 1)
            predicted_image_stack = np.zeros((1, self.xres, self.yres, 1))
            dMdenoised = np.zeros((self.xres, self.yres))

        # re-scale the input data, based on the scaling used when training the model. This info is stored in the model's
        # metadata file, which has been load as the 'mp' directory
        scale_mean = self.mp["noisy_image_stack_mean"]
        scale_std = self.mp["noisy_image_stack_std"]
        noisy_image_stack_rescaled = (noisy_image_stack - scale_mean) / scale_std

        # now run the denoising model
        dMdenoised_raw = self.model.predict(noisy_image_stack_rescaled)

        # undo the rescaling of the predicted diff images to match the original format
        if self.dMraw.ndim == 4:
            sctr = 0
            for t in range(self.tres):
                for k in range(self.zres):
                    thisSlice = dMdenoised_raw[sctr, :, :, 0]
                    thisSlice_unscaled = (thisSlice * scale_std) + scale_mean
                    predicted_image_stack[sctr, :, :, 0] = thisSlice_unscaled
                    # re-format the predicted output as standard nii dimensions
                    dMdenoised[:, :, k, t] = thisSlice_unscaled
                    sctr += 1

        if self.dMraw.ndim == 3:
            for k in range(self.zres):
                thisSlice = dMdenoised_raw[k, :, :, 0]
                thisSlice_unscaled = (thisSlice * scale_std) + scale_mean
                predicted_image_stack[k, :, :, 0] = thisSlice_unscaled
                # re-format the predicted output as standard nii dimensions
                dMdenoised[:, :, k] = thisSlice_unscaled

        if self.dMraw.ndim == 2:
            thisSlice = dMdenoised_raw[0, :, :, 0]
            thisSlice_unscaled = (thisSlice * scale_std) + scale_mean
            predicted_image_stack[0, :, :, 0] = thisSlice_unscaled
            # re-format the predicted output as standard nii dimensions
            dMdenoised = thisSlice_unscaled


        # if we re-sized the input images, we'll undo this for the final output
        if self.rescale_res != self.xres0 or self.rescale_res != self.yres0:

            dMdenoised0 = dMdenoised

            if self.dMraw.ndim == 4:
                dMdenoised = np.zeros((self.xres0, self.yres0, self.zres, self.tres))
                for r in range(self.tres):
                    for k in range(self.zres):
                        thisSlice = dMdenoised0[:, :, k, r]
                        dMdenoised[:, :, k, r] = resize(thisSlice, (self.xres0, self.yres0), anti_aliasing=True)

            if self.dMraw.ndim == 3:
                dMdenoised = np.zeros((self.xres0, self.yres0, self.zres))
                for k in range(self.zres):
                    thisSlice = dMdenoised0[:, :, k]
                    dMdenoised[:, :, k] = resize(thisSlice, (self.xres0, self.yres0), anti_aliasing=True)
                # switch the raw data back to its orgininal size as well
                self.dMraw = self.dMraw0

        # The denoised dataset is now back in the correct, original format
        self.dMdenoised = dMdenoised
        self.modelApplied = True


    def showSlice(self, slice=None, rep=None):
        if self.modelApplied:
            if self.dMraw.ndim == 2:
                pic = np.hstack((np.rot90(self.dMraw), np.rot90(self.dMdenoised)))
                plt.figure()
                plt.imshow(pic, cmap='gray', vmin=-10, vmax=120)

            if self.dMraw.ndim == 3:
                if slice is None:
                    # define default image slice for viewing, if none specified
                    slice = round(self.zres / 2)
                # show example slice
                pic = np.hstack((np.rot90(self.dMraw[:, :, slice]), np.rot90(self.dMdenoised[:, :, slice])))
                plt.figure()
                plt.imshow(pic, cmap='gray', vmin=-10, vmax=120)

            if self.dMraw.ndim == 4:
                if slice is None:
                    # define default image slice for viewing, if none specified
                    slice = round(self.zres / 2)
                if rep is None:
                    # define default repetition for viewing, if none specified
                    rep = 0
                pic = np.hstack((np.rot90(self.dMraw[:, :, slice, rep]), np.rot90(self.dMdenoised[:, :, slice, rep])))
                plt.figure()
                plt.imshow(pic, cmap='gray', vmin=-10, vmax=120)
        else:
            print('** Error: Model has not yet been applied to raw data - please use applyModel() method first **')

    def showTraining(self):
        plt.figure()
        plt.plot(self.epochs, self.val_loss, 'ro-')
        plt.plot(self.epochs, self.loss, 'bo-')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(('validation', 'training'))