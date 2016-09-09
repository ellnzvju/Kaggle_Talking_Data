import numpy as np
import pandas as pd
import sys
import math
from scipy.sparse import csr_matrix, hstack
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.grid_search import GridSearchCV
import xgboost as xgb


class XGBoostClassifier():
    def __init__(self, num_boost_round=10, **params):
        self.clf = None
        self.num_boost_round = num_boost_round
        self.params = params
        self.params.update({'objective': 'multi:softprob'})

    def fit(self, X, y, num_boost_round=None):
        num_boost_round = num_boost_round or self.num_boost_round
        dtrain = xgb.DMatrix(X, label=y)
        self.clf = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=num_boost_round)

    def predict(self, X):
        num2label = {i: label for label, i in self.label2num.items()}
        Y = self.predict_proba(X)
        y = np.argmax(Y, axis=1)
        return np.array([num2label[i] for i in y])

    def predict_proba(self, X):
        dtest = xgb.DMatrix(X)
        return self.clf.predict(dtest)

    def score(self, X, y):
        Y = self.predict_proba(X)
        score = log_loss(y, Y)
        print 'round model score: %f' % score
        return score

    def get_params(self, deep=True):
        return self.params

    def set_params(self, **params):
        if 'num_boost_round' in params:
            self.num_boost_round = params.pop('num_boost_round')
        if 'objective' in params:
            del params['objective']
        if cmp(self.params, params) == -1:
            print ' '
            print 'params: ', params
        self.params.update(params)

        return self



def load_sparse(filename):
    tmp = np.load(filename)
    return csr_matrix((tmp['data'], tmp['indices'], tmp['indptr']), shape= tmp['shape'])


def main():
    global output_location, model_prefix

    Xtrain = load_sparse('WorkSpace/sprm_train.npz')
    Xtest = load_sparse('WorkSpace/sprm_test.npz')
    y = pd.read_csv('WorkSpace/Ytrain.csv')
    # fix random seed for reproducibility
    clf = XGBoostClassifier(
        eval_metric = 'mlogloss',
        num_class = 12,
        nthread = 4,
        silent = 1,
        )

    parameters = {
        'num_boost_round': [100],
        'eta': [0.01],
        'max_depth': [6, 9, 12],
        'subsample': [0.9, 1.0],
        'colsample_bytree': [0.9, 1.0],
        'lambda': [2, 3],
        'alpha': [2]
    }

    clf = GridSearchCV(clf, parameters, n_jobs=1, cv=3, verbose=1)
    clf.fit(Xtrain, y['group'])
    print '-----------------Result---------------------'
    best_parameters, score, _ = min(clf.grid_scores_, key=lambda x: x[1])
    print 'score:', score
    for param_name in sorted(best_parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))

main()
