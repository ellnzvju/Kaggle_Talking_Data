import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.cross_validation import StratifiedKFold, train_test_split # to balance out label
import glob, os
import time
import resource
import xgboost as xgb

output_location = 'Output/'
train_location = output_location + 'stacking_train_10fold/'
test_location = output_location + 'stacking_test_10fold/'
stacking_area = output_location + 'bagging/'



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


    d_test = xgb.DMatrix(test_stacked)


    bagging = 0
    bag_of_predictions = np.zeros((test_stacked.shape[0], 12))
    for x in range(number_of_bagging):

        """ Each bagging iteration """
        print '------------- bagging round %d ------------' % (x + 1)
        test_predict_y = np.zeros((test_stacked.shape[0], 12))

        """ Important to set seed """
        skf = StratifiedKFold(y, n_folds= number_of_folds ,shuffle=True, random_state= None )

        for i, (train_idx, val_idx) in enumerate(skf):
            print 'Fold ', i + 1
            """ Create DMatrix """
            d_train = xgb.DMatrix(train_stacked[train_idx], label=y[train_idx])
            d_val = xgb.DMatrix(train_stacked[val_idx], label=y[val_idx])
            """ Each fold cross validation """
            param = {}
            # use softmax multi-class classification
            param['objective'] = 'multi:softprob'
            # scale weight of positive examples
            param['eta'] = 0.01
            param['max_depth'] = 9
            param['num_class'] = 12
            param['silent'] = 1
            param['alpha'] = 3
            param['eval_metric'] = 'mlogloss'

            watchlist = [(d_train,'train'), (d_val,'validation')]
            num_round = 1000
            bst = xgb.train(param, d_train, num_round, watchlist, early_stopping_rounds=100, verbose_eval= 50 )

            """ Cal test_predict_y """
            t_scores = bst.predict(d_test)
            test_predict_y = test_predict_y + t_scores

        """ CV result """
        test_predict_y = test_predict_y / number_of_folds
        bag_of_predictions = bag_of_predictions + test_predict_y



    bag_of_predictions = bag_of_predictions / number_of_bagging
    filename = 'xgb_stacker_'
    np.save(stacking_area + filename + '10fold_alpha1', bag_of_predictions)


main()
