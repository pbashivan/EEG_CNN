import numpy as np
import scipy.io
import scipy.signal as sig
import pywt
import os.path
import pickle

# checks multichannel EEG (ch x times) for artifacts
def artfct_chk(data, threshold=0, preSize=10, postSize=20):
    flags = np.abs(data) > threshold
    rows, cols = np.nonzero(flags)
    indices = zip(rows, cols)
    for ind in indices:
        pre = max(1, ind[1] - preSize)
        post = min(data.shape[1], ind[1] + postSize)
        flags[ind[0], pre:post] = True
    return flags

# checks multichannel EEG for noisy channel
def noise_chk(data, threshold=10, winSize=0, winShift=0):
    if winSize % 2 == 0:
        winSize += 1
    noise = np.zeros(data.shape)
    stdVals = noise.copy()
    winStart = 0
    winEnd = 1 + winSize

    while winEnd < data.shape[1]:
        stds = np.std(data[:, winStart:winEnd], axis=1)
        if True in (stds > threshold):
            noise[stds > threshold, winStart : winEnd] = True
        stdVals[:, winStart : winEnd] = np.array([stds, ] * (winEnd - winStart)).transpose()
        winStart = winStart + winShift
        winEnd = winStart + winSize
    return noise, stdVals

# Baseline correction by subracting mean of raw EEG
def baseline_correct(data, low_fr=1, high_fr=109, order=100):
    for i, ch in enumerate(data):
        meanData = ch.mean()
        ch = ch - meanData
        # Band-pass filter between low_fr and high_fr
        h = sig.firwin(order, [low_fr, high_fr], nyq=110, pass_zero=False)
        data[i, :] = sig.filtfilt(h, 1, ch)
    return data

# Importing data from matlab cell array
def data_import(datafile, key):
    """ Import data from .mat or .pkl file formats. Converts to float64 type when reading from .mat files.
    key: for .mat file types, name of the variable inside the .mat file.
         for .pkl file types, list of hierarchical variable names.
    """
    data = None

    _, filetype = os.path.splitext(datafile)
    if filetype == '.mat':
        dataTemp = scipy.io.loadmat(datafile)
        dataTempEEG = dataTemp[key]
        data = np.zeros(np.product(dataTempEEG.shape))
        for i, e in enumerate(dataTempEEG.flat):
            data[i] = np.float64(e)
        data = data.reshape(dataTempEEG.shape)
    elif filetype == '.pkl':
        f = open(datafile)
        data = pickle.load(f)
    return data

# Applying discrete WT on each channel
def multch_wavelet(data, wavelet='db4', mode=pywt.MODES.zpd, numLevels=4):
    coeffs = list()
    for i, channel in enumerate(data):
        coeffs.append(pywt.wavedec(channel, wavelet, mode, numLevels))
    return coeffs

# Create Spectrogram as a 2D array
def make_spectro(coeffs, dataShape):
    numLevels = len(coeffs[0]) - 1
    spectrogram = np.zeros(list(dataShape) + [numLevels + 1])
    for c, channel in enumerate(spectrogram):
        for l in np.arange(numLevels + 1):
            if l == 0:
                tileArray = np.repeat(coeffs[c][l], 2 ** (numLevels))
            else:
                tileArray = np.repeat(coeffs[c][l], 2 ** (numLevels - l + 1))
            spectrogram[c, :, l] = tileArray[:dataShape[1]]
    return spectrogram

def normalize_spectro(spectrogram):
    normSpectro = np.zeros(spectrogram.shape)
    for i, ch in enumerate(spectrogram):
        winData = np.abs(ch)
        totalPower = np.repeat(np.sum(winData, axis=1).reshape(-1,1), winData.shape[1], axis=1)
        normSpectro[i, :, :] = winData / totalPower
    return normSpectro


def append_to_dic(dictionary, key, data):
    """Use this function to add columns to a dictionary (for pandas dataframe).
    """
    if dictionary.has_key(key):
        dictionary[key] = np.concatenate((dictionary[key], data), axis=0)
    else:
        dictionary[key] = data