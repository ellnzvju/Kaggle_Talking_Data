import numpy as np
import pandas as pd
import glob, os
from scipy.sparse import csr_matrix, hstack
import sys

def load_sparse(filename):
    tmp = np.load(filename)
    return csr_matrix((tmp['data'], tmp['indices'], tmp['indptr']), shape= tmp['shape'])


def main(argv):
    input_file = argv[0]
    output_file = argv[1]
    Xtest = load_sparse('WorkSpace/sprm_test.npz')
    classes_=['F23-', 'F24-26' ,'F27-28', 'F29-32', 'F33-42', 'F43+', 'M22-', 'M23-26', 'M27-28','M29-31', 'M32-38', 'M39+']
    test = pd.read_csv("Data/gender_age_test.csv", index_col='device_id')

    temp = np.load(input_file)

    output = pd.DataFrame(temp, index = test.index, columns=classes_)
    output.to_csv('Output/submissions/' + output_file, index=True)



if __name__ == "__main__":
   main(sys.argv[1:])
