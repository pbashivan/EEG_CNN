"""Applies k-fold cross validation on the dataset using DBN with training, validation and test sets.
"""

__author__ = "Pouya Bashivan"
__copyright__ = "Copyright 2015, IBM Research YH"
__credits__ = ["Pouya Bashivan"]
__licence__ = "GNU GPL"
__email__ = "poya.bashivan@gmail.com"
__maintainer__ = "Pouya Bashivan"

USEJOBLIB=False

import argparse

import functools
from glob import glob
import logging
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as pl
import multiprocessing
import numpy as np
import os
from os import path
import pandas as pd
import pickle
import random as rndc
from scipy.io import savemat
import scipy.io
from scipy.spatial.distance import pdist
import seaborn as sb

import os
import sys
import timeit
import numpy
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
from logistic_sgd_eeg import LogisticRegression
import mlp_2
from gbrbm import GBRBM
from sklearn.preprocessing import scale
from dbn_4_eeg import DBN


logging.basicConfig(format="[%(module)s:%(levelname)s]:%(message)s")
logger = logging.getLogger(__name__)

# please set this number to no more than the number of cores on the machine you're
# going to be running it on but high enough to help the computation
PROCESSORS = 8
seed = rndc.SystemRandom().seed()
numpy.random.seed(123)


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
    logger.info("Loading data from %s"
                % (data_file))
    
    dataMat = scipy.io.loadmat(data_file, mat_dtype = True)
    data = dataMat['features']
    #numpy.random.shuffle(data)

    logger.info("Data loading complete. Shape is %r" % (data.shape,))
    return data[:, :-1], (data[:, -1]-1)

def reformatInput(data, labels, indices):
    """
    Receives the the indices for train and test datasets.
    Outputs the train, validation, and test data and label datasets.
    """
    # Scale the input into standard normal
    trainIndices = indices[0][len(indices[1]):]
    validIndices = indices[0][:len(indices[1])]
    testIndices = indices[1]

    return [(theano.shared(data[trainIndices, :].astype(theano.config.floatX)), \
         T.cast(labels[trainIndices], 'int32')), \
        (theano.shared(data[validIndices, :].astype(theano.config.floatX)), \
        T.cast(labels[validIndices], 'int32')), \
        (theano.shared(data[testIndices, :].astype(theano.config.floatX)), \
        T.cast(labels[testIndices], 'int32'))]

def create_DBN(datasets, finetune_lr=0.01, pretraining_epochs=10,
             pretrain_lr=0.001, k=1, training_epochs=1000, l1Reg = 0.0001, l2Reg = 0,
             batch_size=10):

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    # numpy random generator
    numpy_rng = numpy.random.RandomState(123)
    print '... building the model'
    # construct the Deep Belief Network
    dbn = DBN(numpy_rng=numpy_rng, n_ins=train_set_x.get_value(borrow=True).shape[1],
              hidden_layers_sizes=[512, 512, 128], l1Reg=l1Reg, l2Reg=l2Reg,
              n_outs=4)

    #########################
    # PRETRAINING THE MODEL #
    #########################
    print '... getting the pretraining functions'
    pretraining_fns = dbn.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size,
                                                k=k)

    print '... pre-training the model'
    start_time = timeit.default_timer()
    ## Pre-train layer-wise
    for i in xrange(dbn.n_layers):
        # go through pretraining epochs
        for epoch in xrange(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                                            lr=pretrain_lr))
            logger.info('Pre-training layer %i, epoch %d, cost %8.2f' % (i, epoch, numpy.mean(c)))

    end_time = timeit.default_timer()
    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing function for the model
    print '... getting the finetuning functions'
    train_fn, validate_model, test_model, test_F_model = dbn.build_finetune_functions(
        datasets=datasets,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )

    print '... finetuning the model'
    # early-stopping parameters
    patience = 100 * n_train_batches  # look as this many examples regardless
    patience_increase = 2.    # wait this much longer when a new best is
                              # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatches before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:

                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses)
                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%'
                    % (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # print a measure of W to check training effect
                print 'minibatch av cost: %f' % minibatch_avg_cost

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = test_model()
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))
                    # print fscore on test set
                    test_fscores = test_F_model()
                    test_fscore = numpy.mean(test_fscores)
                    print(('     epoch %i, minibatch %i/%i, test fscore of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_fscore))
                    

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%, '
            'obtained at iteration %i, '
            'with test performance %f %%'
        ) % (best_validation_loss * 100., best_iter + 1, test_score * 100.)
    )
    print >> sys.stderr, ('The fine tuning code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time)
                                              / 60.))
    return test_score, test_fscore, best_validation_loss

def main(filename, subjectsFilename, out_dir):
    """
    Main function for polyssifier.

    Parameters
    ----------
    source_dir: str
    ksplit: int
    out_dir: str
    """
    # Load input and labels.
    data, labels = load_data(filename)
    mat = scipy.io.loadmat(subjectsFilename, mat_dtype=True)
    subjNumbers = np.squeeze(mat['subjectNum'])     # subject IDs for each trial
    # scale data
    data = scale(data)
    # Make the folds.
        # Leave-Subject-Out cross validation
    fold_pairs = []
    for i in np.unique(subjNumbers):
        ts = subjNumbers == i
        tr = np.squeeze(np.nonzero(np.bitwise_not(ts)))
        np.random.shuffle(tr)
        fold_pairs.append((tr, np.squeeze(np.nonzero(ts))))

    testScores = list()
    testFscores = list()
    validScores = list()

    # Get the scores.
    for i, fold_pair in enumerate(fold_pairs):
        # Reformat the features and labels for theano
        datasets = reformatInput(data, labels, fold_pair)

        # Create DBN
        testScore, testFscore, validScore = create_DBN(datasets)
        testFscores.append(testFscore)
        testScores.append(testScore)
        validScores.append(validScore)

    if out_dir is not None:
        save_path = path.join(out_dir,
                                "results_%.2f.pkl" % (np.mean(testScores)))
        logger.info("Saving classifier to %s" % save_path)
        with open(save_path, "wb") as f:
            pickle.dump((testScores, validScores),f)

    scores = (np.mean(testScores), np.std(testScores))
    logger.info('Test Error rates: %3.2f +- %3.2f' % (scores[0]*100, scores[1]*100))
    scores = (np.mean(testFscores), np.std(testFscores))
    logger.info('Test Fscore rates: %3.2f +- %3.2f' % (scores[0], scores[1]))
    scores = (np.mean(validScores), np.std(validScores))
    logger.info('Valid Error rates: %3.2f +- %3.2f' % (scores[0]*100, scores[1]*100))

if __name__ == "__main__":
    CPUS = multiprocessing.cpu_count()
    if CPUS < PROCESSORS:
        raise ValueError("Number of PROCESSORS exceed available CPUs, "
                         "please edit this in the script and come again!")
    datafile = '/home/pbshivan/Matlab_files/Datasets/WM_features_MSP.mat'
    # outDir = 'Z:\CVPIA\Pouya\SDrive\SkyDrive\Documents\Proposal\Programs\EEG_CNN\Results\\'
    outDir = '/home/pbshivan/Results/'
    subjectsFilename = '/home/pbshivan/trials_subNums.mat'

    logger.setLevel(logging.DEBUG)
    main(datafile, subjectsFilename, out_dir=outDir)
