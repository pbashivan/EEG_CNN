"""This module computes the baseline results by applying various classifiers.
The classifiers used here are nearest neighbors, linear SVM, RBF SVM, decision
tree, random forest, logistic regression, naive bayes, and LDA.
"""

__author__ = "Pouya Bashivan, Sergey Plis"
__copyright__ = "Copyright 2015, Mind Research Network"
__credits__ = ["Pouya Bashivan, Sergey Plis, Devon Hjelm, Alvaro Ulloa"]
__licence__ = "3-clause BSD"
__email__ = "pbshivan@memphis.edu"
__maintainer__ = "Pouya Bashivan"

import logging
from multiprocessing import Pool
from multiprocessing import Array

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

from sklearn.metrics import auc
from sklearn.metrics import roc_curve as rc
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.lda import LDA
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.qda import QDA
from sklearn.preprocessing import scale

import sys


logging.basicConfig(format="[%(module)s:%(levelname)s]:%(message)s")
logger = logging.getLogger(__name__)

# please set this number to no more than the number of cores on the machine you're
# going to be running it on but high enough to help the computation
PROCESSORS = 4
seed = rndc.SystemRandom().seed()
#NAMES = ["Nearest Neighbors", "Linear SVM", "RBF SVM",  "Decision Tree",
#         "Random Forest", "Logistic Regression", "Naive Bayes", "LDA"]
NAMES = ["RBF SVM"]


def make_classifiers(data_shape) :
    """Function that makes classifiers each with a number of folds.

    Returns two dictionaries for the classifiers and their parameters, using
    `data_shape` and `ksplit` in construction of classifiers.

    Parameters
    ----------
    data_shape : tuple of int
        Shape of the data.  Must be a pair of integers.

    Returns
    -------
    classifiers: dict
        The dictionary of classifiers to be used.
    params: dict
        A dictionary of list of dictionaries of the corresponding
        params for each classifier.
    """

    if len(data_shape) != 2:
        raise ValueError("Only 2-d data allowed (samples by dimension).")

    classifiers = {
        "Nearest Neighbors": KNeighborsClassifier(3),
        "Linear SVM": SVC(kernel="linear", C=1, probability=True),
        "RBF SVM": SVC(gamma=2, C=1, probability=True),
        "Decision Tree": DecisionTreeClassifier(max_depth=None,
                                                max_features="auto"),
        "Random Forest": RandomForestClassifier(max_depth=None,
                                                n_estimators=10,
                                                max_features="auto",
                                                n_jobs=PROCESSORS),
        "Logistic Regression": LogisticRegression(),
        "Naive Bayes": GaussianNB(),
        "LDA": LDA()
        }

    params = {
        "Nearest Neighbors": [{"n_neighbors": [1, 5, 10, 20]}],
        "Linear SVM": [{"kernel": ["linear"],"C": [1]}],
        "RBF SVM": [{"kernel": ["rbf"],
                     "gamma": np.arange(0.1, 1, 0.1).tolist() + range(1, 10),
                     "C": np.logspace(-2, 2, 5).tolist()}],
        #"RBF SVM": [{"kernel": ["rbf"],
        #        "gamma": 2,
        #        "C": 50}],
        "Decision Tree": [],
        "Random Forest": [{"n_estimators": range(5,20)}],
        "Logistic Regression": [{"C": np.logspace(0.1, 3, 7).tolist()}],
        "Naive Bayes": [],
        "LDA": [{"n_components": [np.int(0.1 * data_shape[0]),
                                  np.int(0.2 * data_shape[0]),
                                  np.int(0.3 * data_shape[0]),
                                  np.int(0.5 * data_shape[0]),
                                  np.int(0.7 * data_shape[0])]}],
        }

    logger.info("Using classifiers %r with params %r" % (classifiers, params))
    return classifiers, params

def get_score(data, labels, fold_pairs, name, model, param):
    """
    Function to get score for a classifier.

    Parameters
    ----------
    data: array_like
        Data from which to derive score.
    labels: array_like or list
        Corresponding labels for each sample.
    fold_pairs: list of pairs of array_like
        A list of train/test indicies for each fold
        dhjelm(Why can't we just use the KFold object?)
    name: str
        Name of classifier.
    model: WRITEME
    param: WRITEME
        Parameters for the classifier.

    Returns
    -------
    classifier: WRITEME
    fScore: WRITEME
    """
    assert isinstance(name, str)
    logger.info("Classifying %s" % name)

    ksplit = len(fold_pairs)
    if name not in NAMES:
        raise ValueError("Classifier %s not supported. "
                         "Did you enter it properly?" % name)

    # Redefine the parameters to be used for RBF SVM (dependent on
    # training data)

    logger.info("Attempting to use grid search...")
    fScore = []
    accuScores = []
    for i, fold_pair in enumerate(fold_pairs):
        print ("Classifying a %s the %d-th out of %d folds..."
                % (name, i+1, len(fold_pairs)))
        classifier = get_classifier(
            name, model, param, data[fold_pair[0], :])
        fscore, accu = classify(data, labels, fold_pair, classifier)
        fScore.append(fscore)
        accuScores.append(accu)

    return classifier, fScore, accuScores

def get_classifier(name, model, param, data=None):
    """
    Returns the classifier for the model.

    Parameters
    ----------
    name: str
        Classifier name.
    model: WRITEME
    param: WRITEME
    data: array_like, optional

    Returns
    -------
    WRITEME
    """
    assert isinstance(name, str)

    if name == "RBF SVM":
        logger.info("RBF SVM requires some preprocessing."
                    "This may take a while")
        assert data is not None
        #Euclidean distances between samples
        dist = pdist(data, "euclidean").ravel()
        #Estimates for sigma (10th, 50th and 90th percentile)
        sigest = np.asarray(np.percentile(dist,[10,50,90]))
        #Estimates for gamma (= -1/(2*sigma^2))
        gamma = 1./(2*sigest**2)
        #Set SVM parameters with these values
        param = [{"kernel": ["rbf"],
                  "gamma": gamma.tolist(),
                  "C": np.logspace(-2,2,5).tolist()}]
    if name not in ["Decision Tree", "Naive Bayes"]:
        # why 5?
        logger.info("Using grid search for %s" % name)
        model = GridSearchCV(model, param, cv=5, scoring="accuracy",
                             n_jobs=PROCESSORS)
    else:
        logger.info("Not using grid search for %s" % name)
    return model

def classify(data, labels, (train_idx, test_idx), classifier=None):
    """
    Classifies given a fold and a model.

    Parameters
    ----------
    data: array_like
        2d matrix of observations vs variables
    labels: list or array_like
        1d vector of labels for each data observation
    (train_idx, test_idx) : list
        set of indices for splitting data into train and test
    classifier: sklearn classifier object
        initialized classifier with "fit" and "predict_proba" methods.

    Returns
    -------
    WRITEME
    """

    assert classifier is not None, "Why would you pass not classifier?"

    # Data scaling based on training set
    # scaler = StandardScaler()
    # scaler.fit(data[train_idx])
    # data_train = scaler.transform(data[train_idx])
    # data_test = scaler.transform(data[test_idx])
    data_train = data[train_idx]
    data_test = data[test_idx]


    classifier.fit(data_train, labels[train_idx])

    #fpr, tpr, thresholds = rc(labels[test_idx],
    #                          classifier.predict_proba(data_test)[:, 1])
    #return auc(fpr, tpr)
    # Edited by Pouya
    confMat = confusion_matrix(labels[test_idx],
                              classifier.predict(data_test))
    # fscore = f1_score(labels[test_idx],
    #                           classifier.predict(data_test), average='weighted')
    accu = accuracy_score(labels[test_idx],
                             classifier.predict(data_test))
    return accu, confMat
    

def load_data(filename):
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

def main(filename, subjectsFilename, out_dir):
    """
    Main function for polyssifier.

    Parameters
    ----------
    source_dir: str
    out_dir: str
    """
    # Load input and labels.
    data, labels = load_data(filename)
    mat = scipy.io.loadmat(subjectsFilename, mat_dtype=True)
    subjNumbers = np.squeeze(mat['subjectNum'])     # subject IDs for each trial
    # scale data
    data = scale(data, with_std=False)

    # Get classifiers and params.
    classifiers, params = make_classifiers(data.shape)
    global NAMES

    # Make the folds.
    # Leave-Subject-Out cross validation
    fold_pairs = []
    for i in np.unique(subjNumbers):
        ts = subjNumbers == i
        tr = np.squeeze(np.nonzero(np.bitwise_not(ts)))
        np.random.shuffle(tr)
        fold_pairs.append((tr, np.squeeze(np.nonzero(ts))))

    dscore=[]
    accuScores=[]
    for name in NAMES:
        mdl = classifiers[name]
        param = params[name]
        # Get the scores.
        clf, fScores, accuScore = get_score(data, labels,
                                fold_pairs, name,
                                mdl, param)
        
        if out_dir is not None:
            save_path = path.join(out_dir,
                                  name + '.pkl')
            logger.info("Saving classifier to %s" % save_path)
            with open(save_path, "wb") as f:
                pickle.dump(clf,f)
        dscore.append(fScores)
        accuScores.append(accuScore)
        #score[name] = (np.mean(fScores), np.std(fScores))

    # Edited by Pouya
    confMatResults = {name.replace(" ", ""): scores for name, scores in zip(NAMES, dscore)}
    # save results from all folds for estimating AUC std
    # with open(path.join(out_dir, 'auc_score.pkl'),'wb') as f:
    #    pickle.dump(score, f)
    # Save all results
    with open(path.join(out_dir, 'confMats.mat'),'wb') as f:
        scipy.io.savemat(f, confMatResults)

    dscore = np.asarray(dscore)
    accuScores = np.asarray(accuScores)

    with open(path.join(out_dir, 'scores.mat'),'wb') as f:
        scipy.io.savemat(f, {'fscores' : dscore, 
                             'accuScores' : accuScores})
    print 'Average accuracy: {0}'.format(np.mean(accuScores))
    print 'Average fscore: {0}'.format(np.mean(dscore))
    print 'Sum support vecs: {0}'.format(np.sum(clf.best_estimator_.n_support_))
    # pl.figure(figsize=[10,6])
    # ax=pl.gca()
    # ds = pd.DataFrame(dscore.T, columns=NAMES)
    # ds_long =pd.melt(ds)
    # sb.barplot(x='variable', y='value', data=ds_long, palette="Paired")
    # ax.set_xticks(np.arange(len(NAMES)))
    # ax.set_xticklabels(NAMES, rotation=30)
    # ax.set_ylabel("classification AUC")
    # #ax.set_title("Using features: "+str(action_features))
    # pl.subplots_adjust(bottom=0.18)
    # if out_dir is not None:
    #     # change the file you're saving it to
    #     pl.savefig(path.join(out_dir, "classifiers.png"))
    # else:
    #     pl.show(True)

if __name__ == "__main__":
    CPUS = multiprocessing.cpu_count()
    if CPUS < PROCESSORS:
        raise ValueError("Number of PROCESSORS exceed available CPUs, "
                         "please edit this in the script and come again!")
    # datafile = 'Z:\CVPIA\Pouya\SDrive\SkyDrive\Documents\Proposal\Programs\Classifier\Datasets\WM_features_MSP.mat'
    datafile = '/home/pbshivan/Matlab_files/Datasets/WM_features_MSP.mat'
    # outDir = 'Z:\CVPIA\Pouya\SDrive\SkyDrive\Documents\Proposal\Programs\EEG_CNN\Results\\'
    outDir = '/home/pbshivan/Results/'
    subjectsFilename = 'trials_subNums'

    logger.setLevel(logging.DEBUG)
    main(datafile, subjectsFilename, out_dir=outDir)
