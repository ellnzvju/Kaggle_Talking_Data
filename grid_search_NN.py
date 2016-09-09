import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.metrics import log_loss
import glob, os
import time
import resource
from first_level_model_list import model_list
from keras.wrappers.scikit_learn import KerasClassifier
from neural_network_model_catalog import neural_network_catalog
from sklearn.grid_search import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils.np_utils import to_categorical
#tensor flow
#import keras.backend.tensorflow_backend as K

log_location = './Logs/'
output_location = 'Output/'

def load_sparse(filename):
    tmp = np.load(filename)
    return csr_matrix((tmp['data'], tmp['indices'], tmp['indptr']), shape= tmp['shape'])


def relu_baseline(hn1=16, hn2= 32, dp = 0.2, opt = 'adadelta'):
    model = Sequential()
    model.add(Dense(hn1, input_dim= 21396, init='normal', activation='relu'))
    model.add(Dropout(dp))
    model.add(Dense(hn2, init='normal', activation='relu'))
    model.add(Dense(12, init='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])  #logloss
    return model

def prelu_extension(hn1 = 150, hn2 = 50, dp1= 0.4, dp2= 0.2, opt = 'adadelta'):
    model = Sequential()
    model.add(Dense(hn1, input_dim=nb_input, init='normal'))
    model.add(PReLU())
    model.add(Dropout(dp1))
    model.add(Dense(hn2, input_dim=nb_input, init='normal'))
    model.add(PReLU())
    model.add(Dropout(dp2))
    model.add(Dense(12, init='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])  #logloss
    return model



Xtrain = load_sparse('WorkSpace/sprm_train.npz')
Xtest = load_sparse('WorkSpace/sprm_test.npz')
y = pd.read_csv('WorkSpace/Ytrain.csv')
y = y['group']

print 'train data features: %d' % (Xtrain.shape[1])
print 'test data size: %d' % Xtest.shape[0]


y_dummy = to_categorical(y)

def random_gridsearch(iters, params, bfn):
    model = KerasClassifier(build_fn=bfn)
    grid = RandomizedSearchCV(model, param_distributions=params, n_iter=iters, n_jobs=1, cv=10, verbose=1, random_state=10)
    grid_result = grid.fit(Xtrain.toarray(), y_dummy)

    print '----- Best relu -----'
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    for params, mean_score, scores in grid_result.grid_scores_:
        print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))


params = {'nb_epoch': [10, 15, 20, 25],
'batch_size': [64, 128, 256],
'hn1': [64, 100, 225],
'hn2': [64, 100, 225],
'dp': [0.2, 0.3, 0.35, 0.5],
'opt': ['adagrad', 'adadelta', 'adam']
}

random_gridsearch(333, params, relu_baseline)

params = {'nb_epoch': [8, 10, 15, 20],
'batch_size': [128, 256, 400],
'hn1': [100, 150, 225, 300, 625, 725, 900],
'hn2': [50, 75, 100, 150, 225, 300],
'dp1': [0.1, 0.2, 0.25, 0.3, 0.35, 0.5],
'dp2': [0.1, 0.2, 0.25, 0.3, 0.35, 0.5],
'opt': ['adagrad', 'adadelta', 'adam']
}

random_gridsearch(333, params, prelu_extension)
