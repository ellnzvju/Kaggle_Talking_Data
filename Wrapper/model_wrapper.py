import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
import xgboost as xgb
from keras.callbacks import EarlyStopping
from scipy.sparse import issparse


class BaseWrapper(object):

    def __init__(self, name, build_fn, n_input, n_output, **kwargs):
        self.name = name
        self.build_fn = build_fn
        self.n_input = n_input
        self.n_output = n_output
        #self.model = self.build_fn(n_input, n_output, **kwargs)
        self.kwargs = kwargs

    def fit(self, X, y, **kwargs):
        raise NotImplementedError()

    def predict(self, t):
        raise NotImplementedError()

    def predict_proba(self, t):
        raise NotImplementedError()


class KerasClassifier(BaseWrapper):

    ''' batch generator for randomization support sparse
            - has yet implement predict (Since we dont use it normally)
    '''

    def __init__(self, name, build_fn, n_input, n_output, **kwargs):
        super(KerasClassifier, self).__init__(name, build_fn, n_input, n_output, **kwargs)
        self.model = self.build_fn(n_input, n_output, **kwargs)

    def batch_generator(self, X, y, batch_size, shuffle):
        number_of_batches = np.ceil(X.shape[0]/batch_size)
        counter = 0
        sample_index = np.arange(X.shape[0])
        is_sparse = issparse(X)
        if shuffle:
            np.random.shuffle(sample_index)
        while True:
            batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
            X_batch = []
            if( is_sparse ):
                X_batch = X[batch_index,:].toarray()
            else:
                X_batch = X[batch_index,:]
            y_batch = y[batch_index]
            counter += 1
            yield X_batch, y_batch
            if (counter == number_of_batches):
                if shuffle:
                    np.random.shuffle(sample_index)
                counter = 0


    def batch_generatorp(self, X, batch_size):
        number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
        counter = 0
        sample_index = np.arange(X.shape[0])
        is_sparse = issparse(X)
        while True:
            batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
            X_batch = []
            if( is_sparse ):
                X_batch = X[batch_index,:].toarray()
            else:
                X_batch = X[batch_index,:]
            counter += 1
            yield X_batch
            if (counter == number_of_batches):
                counter = 0

    def predict_proba(self, t):
        if( issparse(t) ):
            return self.model.predict_proba(t.toarray(), verbose = 0)
        else:
            return self.model.predict_proba(t, verbose = 0)

    def fit(self, X, y, **kwargs):
        batch_size = 32
        shuffle = False
        nb_epoch = 15
        val_x = []
        val_y = []

        for key, value in self.kwargs.iteritems():
            if key == 'batch_size':
                batch_size = value
            elif key == 'shuffle':
                shuffle = value
            elif key == 'nb_epoch':
                nb_epoch = value

        if 'Validation' in kwargs:
            val_x = []
            if(issparse(kwargs['Validation'][0])):
                val_x = kwargs['Validation'][0].todense()
            else:
                val_x = kwargs['Validation'][0]
            val_y = kwargs['Validation'][1]
            val_y = to_categorical(val_y.tolist(), 12)

        y_dummy = to_categorical(y.tolist())

        if 'Validation' in kwargs:
            fit= self.model.fit_generator(generator=self.batch_generator(X, y_dummy, batch_size, shuffle),
                             nb_epoch= nb_epoch,
                             samples_per_epoch= X.shape[0],
                             validation_data=(val_x, val_y),
                             #callbacks=[EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='auto')],
                             verbose=1)
        else:
            fit= self.model.fit_generator(generator=self.batch_generator(X, y_dummy, batch_size, shuffle),
                             nb_epoch= nb_epoch,
                             samples_per_epoch= X.shape[0],
                             verbose=1)
        return fit


class XgboostClassifier(BaseWrapper):

    def __init__(self, name, build_fn, n_input, n_output, **kwargs):
        super(XgboostClassifier, self).__init__(name, build_fn, n_input, n_output, **kwargs)
        self.params = {}
        self.rounds = 100
        self.early_stop = 20
        for key, value in self.kwargs.iteritems():
            if key == 'num_round':
                self.rounds = value
            elif key == 'early_stopping_rounds':
                self.early_stop = value
            else:
                self.params[key] = value
        self.params['num_class'] = n_output

    def fit(self, X, y, **kwargs):
        d_train = xgb.DMatrix(X, label=y)
        watchlist  = [(d_train,'train')]

        if 'Validation' in kwargs:
            d_val = xgb.DMatrix(kwargs['Validation'][0],label=kwargs['Validation'][1])
            watchlist.append((d_val,'validation'))

        self.model = xgb.train(self.params, d_train, num_boost_round=self.rounds, evals=watchlist, verbose_eval=50, early_stopping_rounds=self.early_stop)

    def predict_proba(self, t):
        d_test = xgb.DMatrix(t)
        return self.model.predict(d_test)
