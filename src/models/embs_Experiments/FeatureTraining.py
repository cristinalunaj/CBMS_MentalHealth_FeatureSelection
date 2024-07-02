"""
    Script to train the models (MLP/SVC...) with embeddings
	author: Cristina Luna.
"""


import os
import sys
import argparse
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
import umap
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from scipy.special import softmax








def get_classifier(model, param, seed = 2020, get_posteriors=False):
    """
    Process dataframe to split features and labels to train the model. It returns the features, the labels and the name of the
    videos in 3 different dataframes.

    :param model:[int] Number of the model to use. 1-SVC / 2- Logistic Regression / 3- ridgeClassifier /4-perceptron
    / 5-NuSVC / 6-LinearSVC / 7-knn / 8-NearestCentroid / 9- DecrissionTree / 10- RandomForest / 11 - MLP')
    :param param:[str] Parameter of the model: C in SVC / C in Logistic Regression / alpha in ridgeClassifier / alpha in perceptron
    / nu in NuSVC / C in LinearSVC / k in knn / None in NearestCentroid / min_samples_split in DecrissionTree / n_estimators in RandomForest / hidden_layer_sizes in MLP
    :param seed:[int] Seed to initialize the random seed generators
    """
    if model == 1:
        print("SVC ")
        classifier = SVC(random_state=seed, C=float(param), probability=get_posteriors)
    elif model == 2:
        print("LOGISTIC REGRES. ")
        classifier = LogisticRegression(random_state=seed, max_iter=10000, C=float(param))
    elif model == 3:
        print("RIDGE CLASSIF. ")
        classifier = RidgeClassifier(random_state=seed, alpha=float(param))
    elif model == 4:
        classifier = Perceptron(random_state=seed, alpha=float(param))
    elif model == 5:
        print("NU SVC ")
        classifier = NuSVC(random_state=seed, nu=float(param))
    elif model == 6:
        print("LINEAR SVC")
        if(get_posteriors):
            classifier = SVC(kernel='linear', random_state=seed, max_iter=10000, C=float(param), probability=get_posteriors)
        else:
            classifier = LinearSVC(random_state=seed, max_iter=10000, C=float(param))
    elif model == 7:
        print("KNN")
        classifier = KNeighborsClassifier(n_neighbors=int(param))
    elif model == 8:
        print("NEAREST CENTROID")
        classifier = NearestCentroid()
    elif model == 9:
        print("DECISSION TREE")
        classifier = sklearn.tree.DecisionTreeClassifier(random_state=seed, min_samples_split=int(param))
    elif model == 10:
        print("RANFOM FOREST")
        classifier = RandomForestClassifier(random_state=seed, n_estimators=int(param))
    elif model == 11:
        print("MLP")
        classifier = MLPClassifier(random_state=seed, hidden_layer_sizes=eval(param))  # learning_rate_init=0.05
    else:
        print('error')
    return classifier



def feature_selection(X,y, featureSelector, out_path_featureSelection, param="", seed = 2020,):
    # How to select feature selector:
    # https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/
    if featureSelector == 1:
        print("ANOVA-F (statistic method) ")
        f_statistic, p_values = f_classif(X,y) # The highest, the best
        # save output:
        df_FS = pd.DataFrame()
        df_FS["F-stat"] = f_statistic
        df_FS["p_values"] = p_values
        df_FS.to_csv(os.path.join(out_path_featureSelection, "statistics_ANOVA-F.csv"), sep=";", index=True)
        # define feature selection
        fs = SelectKBest(score_func=f_classif, k=param)
    else:
        print("to implement")
    return fs





