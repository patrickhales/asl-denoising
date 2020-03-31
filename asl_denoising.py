# Script for implementing a deep-learning-based denoising model on ASL datasets
# Version 1.0, created 31/03/2020
# Author: Patrick Hales

import numpy as np
from matplotlib import pyplot as plt
import nibabel as nib
import os
from keras.models import load_model
from skimage.transform import resize
import pickle


# ------------------- Default Parameters ----------------------------------------------------------------------------------
# Enter the full path to the noisy ASL difference images (dM). These can be either 2D or 3D nifti datasets. An example 3D
# data set (dMnoisy.nii.gz) is included in this package for demonstration purposes.
dMRawFile = 'dMnoisy.nii.gz'

# Enter the deep learning model you wish to apply. The fully trained denoising autoencoder model described in Hales et al.
# (see manuscript_github.pdf in this repository) is included in this package, and is the default model choice
modelFile = 'ADmodel_skipconnections_N_30960_BS_100_EP_100_stackNorm.h5'
# Also define the linked pickle file which contains metadata for the modelFile. The default is included:
modelMetaData = 'ADmodel_skipconnections_N_30960_BS_100_EP_100_stackNorm.pkl'

# Enter the orientation of the noisy images. Training was performed using axial scans, but the model should still work with
# sagittal and coronal scans
orientation = 'axial'           # options: 'axial', 'sagittal', 'coronal'
# ----------------------------------------------------------------------------------------------------------------------

class Dae:
    """
    Class used to implement the trained Denoising Autoencoder model on new data

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
    """
    def __init__(self, dMRawFile=dMRawFile, modelFile=modelFile, modelMetaData=modelMetaData):
        self.dMRawFile = dMRawFile
        self.modelFile = modelFile
        self.modelMetaData = modelMetaData
        self.orientation = orientation

        # define the name for the denoised output file. This will be the same name as the input file, appended with '_dae'
        self.basefolder, self.rawfile = os.path.split(self.dMRawFile)
        rawfile_noextension = self.rawfile.split(".")[0]
        self.outFile = rawfile_noextension + '_dae.nii.gz'
        # define the filepath for the output
        self.outFileFullPath = os.path.join(self.basefolder, self.outFile)

        # open the raw dM file and extract the data
        self.nii = nib.load(dMRawFile)
        self.dMraw = self.nii.get_data().astype('float32')

        # get info about the structure of the raw data
        if self.dMraw.ndim == 3:
            [self.xres, self.yres, self.zres] = self.nii.shape
        else:
             print('Error: input data must consist of a 3D array')


    def applyModel(self):
        # load the keras model
        self.model = load_model(self.modelFile)
        # load the model meta data
        f = open(self.modelMetaData, 'rb')
        self.mp = pickle.load(f)
        f.close()

        # the model is expecting input images of size 128 x 128. If this is not the case, re-size the input
        self.rescale_res = 128
        self.xres0 = self.xres
        self.yres0 = self.yres

        if self.xres0 != self.rescale_res or self.yres0 != self.rescale_res:
            self.dMraw0 = self.dMraw
            self.xres = self.rescale_res
            self.yres = self.rescale_res

            self.dMraw = np.zeros((self.rescale_res, self.rescale_res, self.zres))
            for k in range(self.zres):
                thisSlice = self.dMraw0[:, :, k]
                self.dMraw[:, :, k] = resize(thisSlice, (self.rescale_res, self.rescale_res), anti_aliasing=True)

        # Tidy the input data to remove outliers
        self.dMraw[np.isnan(self.dMraw)] = 0.0
        self.dMraw[np.isinf(self.dMraw)] = 0.0
        dat_test_vector = self.dMraw.flatten()
        uthr = np.percentile(dat_test_vector, 99.9)
        lthr = np.percentile(dat_test_vector, 0.1)
        self.dMraw = np.clip(self.dMraw, a_min=lthr, a_max=uthr)

        # re-scale the input data, based on the scaling used when training the model. This info is stored in the model's
        # metadata file, which has been load as the 'mp' directory
        scale_mean = self.mp["noisy_image_stack_mean"]
        scale_std = self.mp["noisy_image_stack_std"]

        # re-order the input noisy data array, as Keras uses the format <image index, xres, yres, channels>
        predicted_image_stack = np.zeros((self.zres, self.xres, self.yres, 1))
        noisy_image_stack = np.zeros((self.zres, self.xres, self.yres, 1))

        for k in range(self.zres):
            a = self.dMraw[:, :, k]
            noisy_image_stack[k, :, :, :] = (a.reshape(self.xres, self.yres, 1)).astype('float32')

        dMdenoised = np.zeros((self.xres, self.yres, self.zres))

        # re-scale the noisy image stack, using the values from the training of the model
        noisy_image_stack_rescaled = (noisy_image_stack - scale_mean) / scale_std

        # now run the denoising model
        dMdenoised_raw = self.model.predict(noisy_image_stack_rescaled)

        # undo the rescaling of the predicted diff images to match the original format
        for k in range(self.zres):
            thisSlice = dMdenoised_raw[k, :, :, 0]
            thisSlice_unscaled = (thisSlice * scale_std) + scale_mean
            predicted_image_stack[k, :, :, 0] = thisSlice_unscaled
            # re-format the predicted output as standard nii dimensions
            dMdenoised[:, :, k] = thisSlice_unscaled

        # if we re-sized the input images, we'll undo this for the final output
        if self.rescale_res != self.xres0 or self.rescale_res != self.yres0:
            dMdenoised0 = dMdenoised
            dMdenoised = np.zeros((self.xres0, self.yres0, self.zres))
            for k in range(self.zres):
                thisSlice = dMdenoised0 [:, :, k]
                dMdenoised[:, :, k] = resize(thisSlice, (self.xres0, self.yres0), anti_aliasing=True)

        # The denoised dataset is now back in the correct format
        self.dMdenoised = dMdenoised

    def saveResults(self, saveFile=None):
        if saveFile is None:
            saveFile = self.outFileFullPath
        nib.save(nib.Nifti1Image(self.dMdenoised, self.nii.affine, self.nii.header), saveFile)

    def showSlice(self, slice=None):
        if slice is None:
            # define default image slice for viewing, if none specified
            slice = round(self.zres / 2)
        # show example slice
        if self.dMraw.ndim == 3:
            pic = np.hstack((np.rot90(self.dMraw[:, :, slice]), np.rot90(self.dMdenoised[:, :, slice])))
            plt.imshow(pic, cmap='gray', vmin=-10, vmax=120)



