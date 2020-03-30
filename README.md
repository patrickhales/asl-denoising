# asl-denoising
**Combined Denoising and Suppression of Transient Artefacts in Arterial Spin Labelling MRI Using Deep Learning**

This repository contains the Python code needed to create the denoising autoencoder (DAE) model described in "*Combined Denoising and Suppression of Transient Artefacts in Arterial Spin Labelling MRI Using Deep Learning*" by Hales *et al.* Note this paper is currently under review and the model is subject to change. Full details are given in the *manuscript_github.pdf* file.

Current model architecture:
![image](https://user-images.githubusercontent.com/24695126/77834561-882fd000-713d-11ea-8ada-b4eef7958751.png)

**Figure 1**. Architecture of the denoising auto-encoder model, with an example low-SNR, single repetition dMraw image (left), and the corresponding high-SNR dMmean image (right; same axial slice averaged over 10 repetitions). A total of 30,960 image pairs were used to train and validate the model. Image dimensions are shown for each step, along with the number of filter layers used. Skip connections are illustrated as horizontal lines, convolution operations (with subsequent ReLU activation) as green arrows, and max-pooling / up-sampling operations as red / purple arrows respectively. 


