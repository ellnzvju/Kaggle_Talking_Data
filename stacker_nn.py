import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.cross_validation import StratifiedKFold, train_test_split # to balance out label
import glob, os
import time
import resource
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from keras.layers.advanced_activations import PReLU

output_location = 'Output/'
train_location = output_location + 'stacking_train/'
test_location = output_location + 'stacking_test/'

def build_model(input_size):
    model = Sequential()
    model.add(Dense(48, input_dim= input_size, init='glorot_uniform', activation='relu'))
    #model.add(Dropout(0.1))
    model.add(Dense(24, input_dim= input_size, init='glorot_uniform', activation='tanh'))
    model.add(Dense(12, init='glorot_uniform', activation='softmax'))
    # Compile model
    #model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])  #logloss
    #model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='adadelta')

    return model


def batch_generator(X, y, batch_size, shuffle):
    #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:]
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

def batch_generatorp(X, batch_size, shuffle):
    number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :]
        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            counter = 0

def main():
    """ Prepare stack data. load every 1st learner's predictions and stack them together """
    stack = []
    for file in os.listdir(train_location):
        if file.endswith(".npy"):
            temp = np.load(train_location + file)
            stack.append(temp)
            print 'file: %s' % file
    print 'total: %d ' % len(stack)

    test_stack = []
    for file in os.listdir(test_location):
            if file.endswith(".npy"):
                temp = np.load(test_location + file)
                test_stack.append(temp)
                print 'file: %s' % file
    print 'total: %d ' % len(test_stack)


    train_stacked = np.hstack(stack)
    test_stacked = np.hstack(test_stack)
    features = train_stacked.shape[1]

    del stack, test_stack

    number_of_folds = 5
    number_of_bagging = 1

    y = pd.read_csv('WorkSpace/Ytrain.csv')
    y = y['group']

    skf = StratifiedKFold(y, n_folds= number_of_folds ,shuffle=True)

    y_dummy = to_categorical(y.tolist())

    train_predict_y = np.zeros((len(y), 12))
    test_predict_y = np.zeros((test_stacked.shape[0], 12))

    test_predict_list = []
    for i, (train_idx, val_idx) in enumerate(skf):
        """ Each fold iteration """
        print '------------- fold round %d ------------' % i
        """ Each fold cross validation """
        model = build_model(features)

        fit= model.fit_generator(generator=batch_generator(train_stacked[train_idx], y_dummy[train_idx], 128, True),
                         nb_epoch=100,
                         validation_data=(train_stacked[val_idx], y_dummy[val_idx]),
                         samples_per_epoch=train_stacked[train_idx].shape[0], verbose=1
                         )

        scoring = model.predict_proba(train_stacked[val_idx])
        train_predict_y[val_idx] = scoring
        l_score = log_loss(y[val_idx], scoring)
        print '    Fold %d score: %f' % (i, l_score)
        """test stack """
        tresult = model.predict_proba(test_stacked)
        test_predict_y = test_predict_y + tresult

    print 'train prediction...'
    l_score = log_loss(y, train_predict_y)
    print 'Final Fold score: %f' % (l_score)

    print 'test prediction...'
    test_predict_y = test_predict_y / number_of_folds

    filename = 'neural_networ_stacked_'
    np.save(output_location + filename + 'test', test_predict_y)


main()
