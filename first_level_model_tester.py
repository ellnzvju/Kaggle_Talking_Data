import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.metrics import log_loss
import glob, os
import time
import resource
from first_level_model_list import model_list
from Wrapper.model_wrapper import BaseWrapper
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils.np_utils import to_categorical
from first_level_model_list import model_list
from Wrapper.model_wrapper import BaseWrapper
from sklearn.decomposition import PCA

log_location = './Logs/'
output_location = 'Output/'
train_location = output_location + 'first_level_train/'
test_location = output_location + 'first_level_test/'

def load_sparse(filename):
    tmp = np.load(filename)
    return csr_matrix((tmp['data'], tmp['indices'], tmp['indptr']), shape= tmp['shape'])

Xtrain = load_sparse('WorkSpace/sprm_train_v2.npz')
Xtest = load_sparse('WorkSpace/sprm_test_v2.npz')



y = pd.read_csv('WorkSpace/Ytrain.csv')
y = y['group']

print 'train data features: %d' % (Xtrain.shape[1])
print 'test data features: %d' % Xtest.shape[1]


def learning(model_unique_id, number_of_folds= 10):
    print 'Model: %s' % model_unique_id
    np.random.seed(222)
    """ Each model iteration """
    train_predict_y = np.zeros((len(y), 12))
    test_predict_y = np.zeros((Xtest.shape[0], 12))
    ll = 0.
    #777 is good
    skf = StratifiedKFold(y, n_folds= number_of_folds, shuffle=True, random_state= 222)
    """ Each fold cross validation """
    for i, (train_idx, val_idx) in enumerate(skf):
        print 'Fold ', i + 1
        model =  model_list.ModelLookUp(model_unique_id)
        model.fit(Xtrain[train_idx], y[train_idx], **{'Validation': (Xtrain[val_idx], y[val_idx])})

        """ Predict test for each fold """
        tresult = model.predict_proba(Xtest)
        test_predict_y = test_predict_y + tresult
        """ Predict train for each fold """
        scoring = model.predict_proba(Xtrain)
        print 'fold %d : %f', (i, log_loss(y, scoring))
        train_predict_y = train_predict_y + scoring


    print 'average val log_loss: %f' % (ll / number_of_folds)
    """ Train Whole Data """
    print 'test prediction...'
    test_predict_y = test_predict_y / number_of_folds
    train_predict_y = train_predict_y / number_of_folds

    l_score = log_loss(y, train_predict_y)
    print 'Final score: %f' % l_score

    return train_predict_y, test_predict_y




p_train , p_test = learning('xgb_gblinear_high_col',number_of_folds= 5)

filename = 'xgb_leaky_222_'
np.save(train_location + filename, p_train)
np.save(test_location + filename, p_test)
