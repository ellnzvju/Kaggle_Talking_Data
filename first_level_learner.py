import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.metrics import log_loss
import glob, os, sys
import time
import resource
from first_level_model_list import model_list
from Wrapper.model_wrapper import BaseWrapper
import json

nb_class = 0
Xtrain = []
Xtest = []
y = []
train_location = ''
test_location = ''

def load_sparse(filename):
    tmp = np.load(filename)
    return csr_matrix((tmp['data'], tmp['indices'], tmp['indptr']), shape= tmp['shape'])


def load_features(features):
    stack = []
    for f,t in features:
        if t == 'sparse':
            stack.append(load_sparse(f))
        else:
            stack.append(np.load(f))

    return hstack(stack, format='csr')


def load_target():
    y = pd.read_csv('WorkSpace/Ytrain.csv')
    y = y['group']
    return y

def learning(model_unique_id, number_of_folds= 10, seed = 42):
    print 'Model: %s' % model_unique_id

    """ Each model iteration """
    train_predict_y = np.zeros((len(y), nb_class))
    test_predict_y = np.zeros((Xtest.shape[0], nb_class))
    ll = 0.
    """ Important to set seed """
    skf = StratifiedKFold(y, n_folds= number_of_folds ,shuffle=True, random_state= seed )
    """ Each fold cross validation """
    for i, (train_idx, val_idx) in enumerate(skf):
        print 'Fold ', i + 1
        model =  model_list.ModelLookUp(model_unique_id)
        model.fit(Xtrain[train_idx], y[train_idx], **{'Validation': (Xtrain[val_idx], y[val_idx])})
        scoring = model.predict_proba(Xtrain[val_idx])
        """ Out of fold prediction """
        train_predict_y[val_idx] = scoring
        l_score = log_loss(y[val_idx], scoring)
        ll += l_score
        print '    Fold %d score: %f' % (i + 1, l_score)

    print 'average val log_loss: %f' % (ll / number_of_folds)
    """ Fit Whole Data and predict """
    print 'training whole data for test prediction...'
    model =  model_list.ModelLookUp(model_unique_id)
    model.fit(Xtrain, y)
    test_predict_y = model.predict_proba(Xtest)

    filename = model_unique_id + '_' + number_of_folds + 'fold'
    np.save(train_location + filename + '_train' , train_predict_y)
    np.save(test_location + filename + '_test', test_predict_y)


if __name__ == "__main__":
    config_file = sys.argv[1]
    print 'load config file:' , config_file
    config = {}

    with open(config_file) as data_file:
        config = json.load(data_file)

    nb_class = config['n_class']
    np.random.seed(config['seed'])
    train_location = config['train_output']
    test_location = config['test_output']

    print 'prediction class: ' , nb_class
    print 'seed: ', config['seed']
    print 'train output: ', train_location
    print 'test output: ', test_location
    print 'number of folds: ', config['n_folds']

    Xtrain = load_features(config['features']['train'])
    Xtest =  load_features(config['features']['test'])
    y = load_target()

    print 'train data features: %d' % (Xtrain.shape[1])
    print 'test data features: %d' % Xtest.shape[1]
    model_list.nb_input = Xtrain.shape[1]

    for model in config['models']:
        learning(model,number_of_folds= config['n_folds'], seed= config['seed'])
