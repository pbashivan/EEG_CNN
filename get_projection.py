__author__ = 'cvpia'
# import sys
# print sys.path
# sys.path.append('/Users/pouyabashivan/anaconda/lib/python2.7/site-packages/caffe/python')
# sys.path.append('/Users/pouyabashivan/anaconda/lib')
# import caffe
import logging
import numpy as np
np.random.seed(1234)
import scipy.io
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as pl
sb.set_style('white')
sb.set_context('talk')
sb.set(font_scale=2)


import theano
import theano.tensor as T
import lasagne
# from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Conv2DLayer, MaxPool2DLayer, InputLayer
from lasagne.layers import DenseLayer, ElemwiseMergeLayer, FlattenLayer
from lasagne.layers import ConcatLayer, ReshapeLayer, get_output_shape
from lasagne.layers import Conv1DLayer, DimshuffleLayer, LSTMLayer, SliceLayer

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from data_processes import append_to_dic

logging.basicConfig(format="[%(module)s:%(levelname)s]:%(message)s")
logger = logging.getLogger(__name__)

filename = 'EEG_images_32_timeWin'
subjectsFilename = 'trials_subNums'
saved_pars_filename = 'weigths_lasg1.npz'
num_epochs = 30
imSize = 32
batch_size = 20
nb_classes = 4
numTimeWin = 7
GRAD_CLIP = 100     # Clipping value for gradient clipping in LSTM


def showimage(im):
    if im.ndim == 3:
        im = im[:, :, ::-1]
    pl.imshow(im, vmin=0, vmax=255)
    # pl.imshow(im)

# take an array of shape (n, height, width) or (n, height, width, channels)
#  and visualize each (height, width) thing in a grid of size approx. sqrt(n)
#  by sqrt(n)
def vis_square(data, padsize=1, padval=0):
    # data -= data.min()
    # data /= data.max()

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    showimage(data)

def load_vector_data(filename):
    """
    Loads the data from multiple sources if provided.

    Parameters
    ----------
    source_dir: str
    data_pattern: str

    Returns
    -------
    data: array_like
    """
    logger.info("Loading data from %s"
                % (filename))

    dataMat = scipy.io.loadmat(filename, mat_dtype = True)
    data = dataMat['features']

    logger.info("Data loading complete. Shape is %r" % (data.shape,))
    return data[:, :-1], data[:, -1]-1

def build_cnn(input_var=None, W_init=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.
    # Input layer, as usual:
    weights = []        # Keeps the weights for all layers
    layers = dict()
    count = 0
    if W_init is None:
        W_init = [lasagne.init.GlorotUniform()] * 7

    network = InputLayer(shape=(None, 3, imSize, imSize),
                                        input_var=input_var)

    # CNN Group 1
    network = Conv2DLayer(network, num_filters=32, filter_size=(3, 3),
                          W=W_init[count], pad='same')
    count += 1
    weights.append(network.W)
    network = Conv2DLayer(network, num_filters=32, filter_size=(3, 3),
                          W=W_init[count], pad='same')
    count += 1
    weights.append(network.W)
    network = Conv2DLayer(network, num_filters=32, filter_size=(3, 3),
                          W=W_init[count], pad='same')
    count += 1
    weights.append(network.W)
    network = Conv2DLayer(network, num_filters=32, filter_size=(3, 3),
                          W=W_init[count], pad='same')
    count += 1
    weights.append(network.W)
    network = MaxPool2DLayer(network, pool_size=(2, 2))
    layers['conv1_out'] = network

    # CNN Group 2
    network = Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                          W=W_init[count], pad='same')
    count += 1
    weights.append(network.W)
    network = Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                          W=W_init[count], pad='same')
    count += 1
    weights.append(network.W)
    # network = Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
    #                       W=W_init[count], pad='same')
    # count += 1
    # weights.append(network.W)
    network = MaxPool2DLayer(network, pool_size=(2, 2))
    layers['conv2_out'] = network

    # CNN Group 3
    network = Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                          W=W_init[count], pad='same')
    count += 1
    weights.append(network.W)
    # network = Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
    #                       W=W_init[count], pad='same')
    # count += 1
    # weights.append(network.W)
    # network = Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
    #                       W=W_init[count], pad='same')
    # count += 1
    # weights.append(network.W)
    network = MaxPool2DLayer(network, pool_size=(2, 2))
    layers['conv3_out'] = network

    return network, weights


def build_convpool_lstm(input_vars):
    convnets = []
    W_init = None
    for i in range(numTimeWin):
        if i == 0:
            convnet, W_init = build_cnn(input_vars[i])
        else:
            convnet, _ = build_cnn(input_vars[i], W_init)
        convnets.append(FlattenLayer(convnet))
    # at this point convnets shape is [numTimeWin][n_samples, features]
    # we want the shape to be [n_samples, features, numTimeWin]
    convpool = ConcatLayer(convnets)
    # convpool = ReshapeLayer(convpool, ([0], -1, numTimeWin))

    convpool = ReshapeLayer(convpool, ([0], numTimeWin, get_output_shape(convnets[0])[1]))
    # Input to LSTM should have the shape as (batch size, SEQ_LENGTH, num_features)
    convpool = LSTMLayer(convpool, num_units=128, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh)
    # After LSTM layer you either need to reshape or slice it (depending on whether you
    # want to keep all predictions or just the last prediction.
    # http://lasagne.readthedocs.org/en/latest/modules/layers/recurrent.html
    # https://github.com/Lasagne/Recipes/blob/master/examples/lstm_text_generation.py
    convpool = SliceLayer(convpool, -1, 1)      # Selecting the last prediction
    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    convpool = DenseLayer(lasagne.layers.dropout(convpool, p=.5),
            num_units=512, nonlinearity=lasagne.nonlinearities.rectify)
    # We only need the final prediction, we isolate that quantity and feed it
    # to the next layer.

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    convpool = DenseLayer(lasagne.layers.dropout(convpool, p=.5),
            num_units=4, nonlinearity=lasagne.nonlinearities.softmax)
    return convpool

def build_convpool_mix(input_vars):
    convnets = []
    W_init = None
    for i in range(numTimeWin):
        if i == 0:
            convnet, W_init = build_cnn(input_vars[i])
        else:
            convnet, _ = build_cnn(input_vars[i], W_init)
        convnets.append(FlattenLayer(convnet))
    # at this point convnets shape is [numTimeWin][n_samples, features]
    # we want the shape to be [n_samples, features, numTimeWin]
    convpool = ConcatLayer(convnets)
    # convpool = ReshapeLayer(convpool, ([0], -1, numTimeWin))

    convpool = ReshapeLayer(convpool, ([0], numTimeWin, get_output_shape(convnets[0])[1]))
    reformConvpool = DimshuffleLayer(convpool, (0, 2, 1))

    # input to 1D convlayer should be in (batch_size, num_input_channels, input_length)
    conv_out = Conv1DLayer(reformConvpool, 32, 3)
    conv_out = FlattenLayer(conv_out)
    # Input to LSTM should have the shape as (batch size, SEQ_LENGTH, num_features)
    lstm = LSTMLayer(convpool, num_units=128, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh)
    # After LSTM layer you either need to reshape or slice it (depending on whether you
    # want to keep all predictions or just the last prediction.
    # http://lasagne.readthedocs.org/en/latest/modules/layers/recurrent.html
    # https://github.com/Lasagne/Recipes/blob/master/examples/lstm_text_generation.py
    lstm_out = SliceLayer(convpool, -1, 1)

    # Merge 1D-Conv and LSTM outputs
    dense_input = ConcatLayer([conv_out, lstm_out])
    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    convpool = DenseLayer(lasagne.layers.dropout(dense_input, p=.5),
            num_units=512, nonlinearity=lasagne.nonlinearities.rectify)

        # We only need the final prediction, we isolate that quantity and feed it
    # to the next layer.

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    convpool = DenseLayer(lasagne.layers.dropout(convpool, p=.5),
            num_units=4, nonlinearity=lasagne.nonlinearities.softmax)
    return convpool

if __name__ == '__main__':
    subj_number = 3
    mat = scipy.io.loadmat(subjectsFilename, mat_dtype=True)
    subjNumbers = np.squeeze(mat['subjectNum'])     # subject IDs for each trial

    vector_file = 'Z:\CVPIA\Pouya\SDrive\SkyDrive\Documents\Proposal\Programs\Classifier\Datasets\Time Sliced\WM_features_MSP_7windows.mat'
    # vector_file = '/Users/pouyabashivan/Documents/Onedrive/Documents/Proposal/Programs/Classifier/Datasets/Time Sliced/WM_features_MSP_7windows.mat'
    data, labels = load_vector_data(vector_file)
    # t-SNE projection
    # Raw Data
    # Select data and labels for a specific subject
    index = subjNumbers==subj_number
    # subject_data = data[index]
    # subject_labels = labels[index]
    # # PCA
    # pca = PCA(n_components=50)       # keeping top 3 components
    # PCs = pca.fit_transform(subject_data)
    # # t-SNE
    # model = TSNE(n_components=2, random_state=0)
    # embeddings = model.fit_transform(PCs)
    # dataFrameDic = dict()   # Dictionary for keeping all table information included in DataFrame
    # for cNum, emb in enumerate(embeddings.transpose()):
    #     append_to_dic(dataFrameDic, 'emb'+ str(cNum+1), emb)
    # append_to_dic(dataFrameDic, 'Load', subject_labels)
    # frame = pd.DataFrame(dataFrameDic)
    # grid = sb.FacetGrid(frame, hue='Load')
    # grid.map(pl.scatter, 'emb1', 'emb2', alpha=0.7, s=40)
    # grid.add_legend();pl.show()

    # For network output features
    saved_pars = np.load(saved_pars_filename)
    param_values = [saved_pars['arr_%d' % i] for i in range(len(saved_pars.files))]

    mat = scipy.io.loadmat(filename)
    featureMatrix = mat['featMat']
    labels = mat['labels']
    input_var = T.TensorType('floatX', ((False,) * 5))()        # Notice the () at the end
    target_var = T.ivector('targets')
    network = build_convpool_mix(input_var)
    # network = build_convpool_lstm(input_var)
    lasagne.layers.set_all_param_values(network, param_values)
    layers = lasagne.layers.get_all_layers(network)
    filterNum = 20

    # # Calculating representations (Inputs to the network consisting of 7 frames)
    input_dic = dict(
        {layers[0]: featureMatrix[0, index],
         layers[12]: featureMatrix[1, index],
         layers[24]: featureMatrix[2, index],
         layers[36]: featureMatrix[3, index],
         layers[48]: featureMatrix[4, index],
         layers[60]: featureMatrix[5, index],
         layers[72]: featureMatrix[6, index],
         }
    )

    # Listing the saved layers
    # for i, l in enumerate(layers):
    #     print('Layer{0}: {1}'.format(i, l))

    output = lasagne.layers.get_output(layers[6], inputs=input_dic)
    # representations = output.eval()
    #
    # # PCA
    # PCs = pca.fit_transform(representations)
    # # t-SNE
    # embeddings = model.fit_transform(PCs)
    # dataFrameDic = dict()   # Dictionary for keeping all table information included in DataFrame
    # for cNum, emb in enumerate(embeddings.transpose()):
    #     append_to_dic(dataFrameDic, 'emb'+ str(cNum+1), emb)
    # append_to_dic(dataFrameDic, 'Load', subject_labels)
    # frame = pd.DataFrame(dataFrameDic)
    # grid = sb.FacetGrid(frame, hue='Load')
    # grid.map(pl.scatter, 'emb1', 'emb2', alpha=0.7, s=40)
    # grid.add_legend(); pl.show()

    # Finding the average feature values
    load_index = np.squeeze(np.bitwise_and(index, labels == 4))
    mean_responses = np.mean(featureMatrix[:, load_index], axis=1)
    mean_responses = featureMatrix[:, 1573]
    # pl.ion()
    # pl.figure()
    # for i in range(mean_responses.shape[0]):
    #     pl.subplot(170+i)
    #     pl.imshow(np.swapaxes(np.rollaxis(mean_responses[i], 0, 3), 0, 1))
    #     pl.grid(False)
    #     pl.gca().axes.get_xaxis().set_visible(False)
    #     pl.gca().axes.get_yaxis().set_visible(False)
    # pl.savefig('eeg_frames_subj3_load4_1', dpi=300)

    testSampleNum = 12
    sampleNum = np.nonzero(index)[0][testSampleNum]

    pl.figure(); pl.imshow(np.swapaxes(np.rollaxis(featureMatrix[0, sampleNum], 0, 3), 0, 1)); pl.show()

    output = lasagne.layers.get_output(layers[5], inputs=input_dic)
    data = output.eval()[testSampleNum, :]; pl.figure(); vis_square(data)

    # output = lasagne.layers.get_output(layers[8], inputs=input_dic)
    # data = output.eval()[testSampleNum, :]; pl.figure(); vis_square(data)
    #
    # output = lasagne.layers.get_output(layers[10], inputs=input_dic)
    # data = output.eval()[testSampleNum, :]; pl.figure(); vis_square(data)

    # pl.subplot(221); pl.imshow(np.swapaxes(np.rollaxis(featureMatrix[0, sampleNum], 0, 3), 0, 1))
    # pl.subplot(222); pl.imshow(output.eval()[0, filterNum], cmap='Reds')
    #
    # output = lasagne.layers.get_output(layers[8], inputs=featureMatrix[:, sampleNum])
    # pl.subplot(223); pl.imshow(output.eval()[0, filterNum], cmap='Reds')
    #
    # output = lasagne.layers.get_output(layers[10], inputs=featureMatrix[:, sampleNum])
    # pl.subplot(224); pl.imshow(output.eval()[0, filterNum], cmap='Reds')
    # pl.show()
    # t-SNE projection
    # Learned representations


    # for i in xrange(len(labels)):
    #     output = lasagne.layers.get_output(layers[5], inputs=featureMatrix[:, i])
    #     output.eval()[0, filterNum]