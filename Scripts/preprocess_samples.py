"""
Reads the electrode locations and a matrix of EEG features. Generates
EEG images with three color channels.
Feature matrix is structured as [samples, features]. Features are
power in each frequency band over all electrodes
(theta-1, theta-2, ..., theta-64, alpha-1, ..., alpha-64, ...)
Locations file should contain coordinates for number of electrodes
existing in the features file.
"""
import numpy as np
import matplotlib.pyplot as pl
import scipy.io
from scipy.interpolate import griddata
from scipy.misc import bytescale
from sklearn.preprocessing import scale
from utils import augment_EEG

augment = False                 # Augment data
pca = False                     # Augment using PCA
stdMult = 0.1                   # Standard deviation of added noise
n_components = 2                # Number of components to keep
nGridPoints = 32                # Number of pixels in the image
nColors = 3                     # Number of color channels in the output image
nElectrodes = 64                # Number of electrodes

# Load electrode locations projected on a 2D surface
mat = scipy.io.loadmat('Neuroscan_locs_polar_proj.mat')
locs = mat['proj'] * [1, -1]    # reverse the Y axis to have the front on the top

# Plot Electrode locations
# fig = pl.figure()
# ax = fig.add_subplot(111)
# ax.scatter(locs[:, 0],locs[:, 1])
# ax.set_xlabel('X'); ax.set_ylabel('Y')
# ax.set_title('Polar Projection')
# pl.ion(); pl.show()

# Load power values
mat = scipy.io.loadmat('Z:\CVPIA\Pouya\SDrive\SkyDrive\Documents\Proposal\Programs\Classifier\Datasets\WM_features_MSP.mat')
data = mat['features']
thetaFeats = data[:, :64]
alphaFeats = data[:, 64:128]
betaFeats = data[:, 128:192]

if augment:
    if pca:
        thetaFeats = augment_EEG(thetaFeats, stdMult, pca=True, n_components=n_components)
        alphaFeats = augment_EEG(alphaFeats, stdMult, pca=True, n_components=n_components)
        betaFeats = augment_EEG(betaFeats, stdMult, pca=True, n_components=n_components)
    else:
        thetaFeats = augment_EEG(thetaFeats, stdMult, pca=False, n_components=n_components)
        alphaFeats = augment_EEG(alphaFeats, stdMult, pca=False, n_components=n_components)
        betaFeats = augment_EEG(betaFeats, stdMult, pca=False, n_components=n_components)

labels = data[:, -1]
nSamples = data.shape[0]

# Interpolate the values
grid_x, grid_y = np.mgrid[
                 min(locs[:, 0]):max(locs[:, 0]):nGridPoints*1j,
                 min(locs[:, 1]):max(locs[:, 1]):nGridPoints*1j
                 ]
thetaInterp = np.zeros([nSamples, nGridPoints, nGridPoints])
alphaInterp = np.zeros([nSamples, nGridPoints, nGridPoints])
betaInterp = np.zeros([nSamples, nGridPoints, nGridPoints])

for i in xrange(nSamples):
    thetaInterp[i, :, :] = griddata(locs, thetaFeats[i, :], (grid_x, grid_y),
                                    method='cubic', fill_value=np.nan)
    alphaInterp[i, :, :] = griddata(locs, alphaFeats[i, :], (grid_x, grid_y),
                                    method='cubic', fill_value=np.nan)
    betaInterp[i, :, :] = griddata(locs, betaFeats[i, :], (grid_x, grid_y),
                                   method='cubic', fill_value=np.nan)
    print 'Interpolating {0}/{1}\r'.format(i+1, nSamples),

# Byte scale to 0-255 range and substituting NaN with 0
# thetaInterp[~np.isnan(thetaInterp)] = bytescale(thetaInterp[~np.isnan(thetaInterp)])
thetaInterp[~np.isnan(thetaInterp)] = scale(thetaInterp[~np.isnan(thetaInterp)])
thetaInterp = np.nan_to_num(thetaInterp)
# alphaInterp[~np.isnan(alphaInterp)] = bytescale(alphaInterp[~np.isnan(alphaInterp)])
alphaInterp[~np.isnan(alphaInterp)] = scale(alphaInterp[~np.isnan(alphaInterp)])
alphaInterp = np.nan_to_num(alphaInterp)
# betaInterp[~np.isnan(betaInterp)] = bytescale(betaInterp[~np.isnan(betaInterp)])
betaInterp[~np.isnan(betaInterp)] = scale(betaInterp[~np.isnan(betaInterp)])
betaInterp = np.nan_to_num(betaInterp)

featureMatrix = np.zeros((nColors, nSamples, nGridPoints, nGridPoints))
featureMatrix[0, :, :, :] = thetaInterp
featureMatrix[1, :, :, :] = alphaInterp
featureMatrix[2, :, :, :] = betaInterp
featureMatrix = np.swapaxes(featureMatrix, 0, 1)        # swap axes to have [samples, colors, W, H]

# Save all data into mat file
scipy.io.savemat('EEG_images_32_flattened_locs', {'featMat': featureMatrix,
                                'labels': labels})


## ALL IMAGES SHOULD BE FLIPPED UPSIDE-DOWN ##
## USE np.flipud #############################
ind = 12
pl.figure()
# pl.subplot(321)
# negate the Y-Axis to make the image upside down
# pl.scatter(locs[:, 0], -locs[:,1], s=100, c=thetaFeats[ind, :].T); pl.title('Theta Electrodes')
pl.subplot(221)
pl.imshow(featureMatrix[ind, 0, :, :].T, cmap='Reds', vmin=np.min(featureMatrix[ind]), vmax=np.max(featureMatrix[ind])); pl.title('Theta')
pl.subplot(222)
pl.imshow(featureMatrix[ind, 1, :, :].T, cmap='Greens', vmin=np.min(featureMatrix[ind]), vmax=np.max(featureMatrix[ind])); pl.title('Alpha')
pl.subplot(223)
pl.imshow(featureMatrix[ind, 2, :, :].T, cmap='Blues', vmin=np.min(featureMatrix[ind]), vmax=np.max(featureMatrix[ind])); pl.title('Beta')
pl.subplot(224)
pl.imshow(np.swapaxes(np.rollaxis(featureMatrix[ind, :, :, :], 0, 3), 0, 1), vmin=np.min(featureMatrix[ind]), vmax=np.max(featureMatrix[ind])); pl.title('All')
pl.show()
