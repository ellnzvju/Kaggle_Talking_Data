import numpy as np
import pandas as pd
import glob, os
from sklearn.metrics import log_loss

def load_sparse(filename):
    tmp = np.load(filename)
    return csr_matrix((tmp['data'], tmp['indices'], tmp['indptr']), shape= tmp['shape'])

output_location = 'Output/'

train_location = output_location + 'first_level_train/'
#test_location = output_location + 'first_level_test/'
test_location = output_location + 'second_level_collections/'


ensemble_name = 'ensembled_'
train_size = 74645
test_size = 112071

""" Ensemble any npy file in BaggingOutput """
""" train_ensemble will test result (log_loss and correct) """
def ensemble(train_data=True):
    size = test_size
    directory = test_location

    y = pd.read_csv('WorkSpace/Ytrain.csv')
    y = y['group']

    if train_data:
        size = train_size
        directory = train_location

    files = []
    total_file = 0
    print '------------- Ensemble ------------'
    for file in os.listdir(directory):
        if file.endswith(".npy"):
            print 'file number[%d]: %s' % (total_file, file)
            temp = np.load(directory + file)
            files.append((file,temp))
            total_file += 1


    ensembles = []

    val_score = 0.0
    while True:
        print(chr(27) + "[2J")
        final_result = np.ones((size,12))
        print '--------file list--------'
        for i  in range(len(files)):
            print ' [%d] %s' % (i, files[i][0])

        print '--------Items in batch ------'
        for i in range(len(ensembles)):
            print ' [%d] %s' % (i, ensembles[i][0])
            final_result = final_result * ensembles[i][1]

        if len(ensembles) > 0:
            root = 1./float(len(ensembles))
            final_result = np.power(final_result, root)
        print '-----------------------------'
        if train_data and len(ensembles) > 0:
            val_score = log_loss(y, final_result)
            print 'current score: %f' % val_score

        print 'A - add, D-delete, S-Save, C-clear, M-multiply'
        cmd = raw_input('>')

        if cmd == 'a' or cmd =='A':
            idx = int(raw_input('number>'))
            if idx < len(files) and idx >= 0:
                ensembles.append(files[idx])
        elif cmd == 'd' or cmd =='D':
            idx = int(raw_input('number>'))
            if idx < len(ensembles) and idx >= 0:
                ensembles.pop(idx)
        elif cmd == 'c' or cmd =='C':
            ensembles = []
            final_result = np.zeros((size,12))
        elif cmd == 's' or cmd == 'S':
            ename = raw_input('input name>')
            np.save(output_location + ename, final_result)
        elif cmd == 'aa':
            for x in files:
                ensembles.append(x)



if __name__ == "__main__":
   ensemble(False)
