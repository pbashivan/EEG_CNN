# Covolutional NN for EEG images made out of power spectrum within three bands.
# Uses training, validation and test datasets.
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import scipy.io
np.random.seed(123)

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from keras.layers.normalization import BatchNormalization
from six.moves import range

from sklearn.cross_validation import StratifiedKFold, KFold


'''
    Train a (fairly simple) deep CNN on the CIFAR10 small images dataset.

    GPU run command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py

    It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
    (it's still underfitting at that point, though).

    Note: the data was pickled with Python 2, and some encoding issues might prevent you
    from loading it in Python 3. You might have to load it in Python 2,
    save it in a different format, load it in Python 3 and repickle it.
'''

filename = 'EEG_images'
subjectsFilename = 'trials_subNums'
batch_size = 20
nb_classes = 4
nb_epoch = 40
data_augmentation = True

# shape of the image (SHAPE x SHAPE)
shapex, shapey = 40, 40
# number of convolutional filters to use at each layer
nb_filters = [40, 80]
# level of pooling to perform at each layer (POOL x POOL)
nb_pool = [2, 2]
# level of convolution to perform at each layer (CONV x CONV)
nb_conv = [3, 3]
# the CIFAR10 images are RGB
image_dimensions = 3

def load_data(data_file):
    """
    Loads the data from multiple sources if provided.

    Parameters
    ----------
    source_dir: str

    Returns
    -------
    data: array_like
    """
    print("Loading data from %s" % (data_file))

    dataMat = scipy.io.loadmat(data_file, mat_dtype=True)
    data = dataMat['featMat']
    labels = dataMat['labels']
    # indices = np.random.permutation(labels.shape[1])      # shuffling indices

    print("Data loading complete. Shape is %r" % (dataMat['featMat'].shape,))
    # return data[indices, :, :, :].astype(np.uint8), labels[:, indices].T - 1        # Shuffled indices
    return dataMat['featMat'].astype(np.uint8), dataMat['labels'].T - 1   # Sequential indices


def reformatInput(data, labels, indices):
    """
    Receives the the indices for train and test datasets.
    Outputs the train, validation, and test data and label datasets.
    """

    trainIndices = indices[0]
    validIndices = indices[1][:np.floor(0.5 * indices[1].shape[0]).astype('int32')]
    testIndices = indices[1][np.floor(0.5 * indices[1].shape[0]).astype('int32'):]
    # Shuffling training data
    # shuffledIndices = np.random.permutation(len(trainIndices))
    # trainIndices = trainIndices[shuffledIndices]

    return [(data[trainIndices, :, :, :], np.squeeze(labels[trainIndices]).astype(np.int32)),
            (data[validIndices, :, :, :], np.squeeze(labels[validIndices]).astype(np.int32)),
            (data[testIndices, :, :, :], np.squeeze(labels[testIndices]).astype(np.int32))]

def main():
    # the data, shuffled and split between tran and test sets
    data, labels = load_data(filename)
    mat = scipy.io.loadmat(subjectsFilename, mat_dtype=True)
    subjNumbers = np.squeeze(mat['subjectNum'])     # subject IDs for each trial

    # Creating the folds
    # kf = StratifiedKFold(np.squeeze(labels), n_folds=ksplit, shuffle=True, random_state=123)
    # kf = KFold(labels.shape[0], n_folds=ksplit, shuffle=True, random_state=123)
    # fold_pairs = [(tr, ts) for (tr, ts) in kf]

    # Leave-Subject-Out cross validation
    fold_pairs = []
    for i in np.unique(subjNumbers):
        ts = subjNumbers == i
        tr = np.squeeze(np.nonzero(np.bitwise_not(ts)))
        np.random.shuffle(ts)       # Shuffle indices
        np.random.shuffle(tr)
        fold_pairs.append((tr, np.squeeze(np.nonzero(ts))))


    validScores, testScores = [], []
    for foldNum, fold in enumerate(fold_pairs):
        (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = reformatInput(data, labels, fold)
        print('X_train shape:', X_train.shape)
        print(X_train.shape[0], 'train samples')
        print(X_valid.shape[0], 'valid samples')
        print(X_test.shape[0], 'test samples')

        X_train = X_train.astype("float32", casting='unsafe')
        X_valid = X_valid.astype("float32", casting='unsafe')
        X_test = X_test.astype("float32", casting='unsafe')
        # convert class vectors to binary class matrices
        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_valid = np_utils.to_categorical(y_valid, nb_classes)
        Y_test = np_utils.to_categorical(y_test, nb_classes)

        # Building the network
        model = Sequential()

        model.add(Convolution2D(40, 3, 3, border_mode='full',
                                input_shape=(image_dimensions, shapex, shapey)))
        model.add(Activation('relu'))
        model.add(Convolution2D(40, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # model.add(Convolution2D(80, 3, 3, border_mode='full'))
        # model.add(Activation('relu'))
        # model.add(Convolution2D(80, 3, 3))
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))

        # model.add(Convolution2D(nb_filters[0], image_dimensions, nb_conv[0], nb_conv[0], border_mode='full'))
        # # model.add(BatchNormalization([nb_filters[0], nb_conv[0], nb_conv[0], image_dimensions]))
        # model.add(Activation('relu'))
        # model.add(Convolution2D(nb_filters[0], nb_filters[0], nb_conv[0], nb_conv[0]))
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(poolsize=(nb_pool[0], nb_pool[0])))
        # model.add(Dropout(0.25))
        #
        # model.add(Convolution2D(nb_filters[1], nb_filters[0], nb_conv[0], nb_conv[0], border_mode='full'))
        # model.add(Activation('relu'))
        # model.add(Convolution2D(nb_filters[1], nb_filters[1], nb_conv[1], nb_conv[1]))
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(poolsize=(nb_pool[1], nb_pool[1])))
        # model.add(Dropout(0.25))
        #
        # model.add(Flatten())
        # # the image dimensions are the original dimensions divided by any pooling
        # # each pixel has a number of filters, determined by the last Convolution2D layer
        # model.add(Dense(nb_filters[-1] * (shapex / nb_pool[0] / nb_pool[1]) * (shapey / nb_pool[0] / nb_pool[1]), 1024))
        # # model.add(BatchNormalization([1024]))
        # model.add(Activation('relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(1024, nb_classes))
        # model.add(Activation('softmax'))

        # let's train the model using SGD + momentum (how original).
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd)

        if not data_augmentation:
            print("Not using data augmentation or normalization")

            X_train = X_train.astype("float32", casting='unsafe')
            X_test = X_test.astype("float32", casting='unsafe')
            X_train /= 255.
            X_test /= 255.
            model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True)
            score, accu = model.evaluate(X_test, Y_test, batch_size=batch_size, show_accuracy=True)
            print('Test accuracy:', accu)

        else:
            print("Using real time data augmentation")
            # X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train.flatten(), axis=0)
            # X_valid = (X_valid - np.mean(X_valid, axis=0)) / np.std(X_valid.flatten(), axis=0)
            # X_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test.flatten(), axis=0)
            # X_train = (X_train - np.mean(X_train, axis=0))
            # X_valid = (X_valid - np.mean(X_train, axis=0))
            # X_test = (X_test - np.mean(X_train, axis=0))
            # this will do preprocessing and realtime data augmentation
            datagen = ImageDataGenerator(
                featurewise_center=True,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=False,  # randomly flip images
                vertical_flip=False)  # randomly flip images

            # compute quantities required for featurewise normalization
            # (std, mean, and principal components if ZCA whitening is applied)
            datagen.fit(X_train)
            best_validation_accu = 0

            for e in range(nb_epoch):

                print('-'*40)
                print('Epoch', e)
                print('-'*40)
                print("Training...")
                # batch train with realtime data augmentation
                progbar = generic_utils.Progbar(X_train.shape[0])
                for X_batch, Y_batch in datagen.flow(X_train, Y_train, batch_size=batch_size):
                    score, trainAccu = model.train_on_batch(X_batch, Y_batch, accuracy=True)
                    progbar.add(X_batch.shape[0], values=[("train accuracy", trainAccu)])

                print("Validating...")
                # Validation time!
                progbar = generic_utils.Progbar(X_valid.shape[0])
                epochValidAccu = []
                for X_batch, Y_batch in datagen.flow(X_valid, Y_valid, batch_size=batch_size):
                    score, validAccu = model.test_on_batch(X_batch, Y_batch, accuracy=True)
                    epochValidAccu.append(validAccu)
                    progbar.add(X_batch.shape[0], values=[("validation accuracy", validAccu)])
                meanValidAccu = np.mean(epochValidAccu)
                if meanValidAccu > best_validation_accu:
                    best_validation_accu = meanValidAccu
                    best_iter = e
                    print("Testing...")
                    # test time!
                    progbar = generic_utils.Progbar(X_test.shape[0])
                    epochTestAccu = []
                    for X_batch, Y_batch in datagen.flow(X_test, Y_test, batch_size=batch_size):
                        score, testAccu = model.test_on_batch(X_batch, Y_batch, accuracy=True)
                        epochTestAccu.append(testAccu)
                        progbar.add(X_batch.shape[0], values=[("test accuracy", testAccu)])
                    model.save_weights('weigths_{0}'.format(foldNum), overwrite=True)
            validScores.append(best_validation_accu)
            testScores.append(np.mean(epochTestAccu))
    scipy.io.savemat('cnn_results', {'validAccu': validScores, 'testAccu': testScores})
    print ('Average valid accuracies: {0}'.format(np.mean(validScores)))
    print ('Average test accuracies: {0}'.format(np.mean(testScores)))

if __name__ == '__main__':
    main()