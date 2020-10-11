from ml_tools import CosineEstimator, Similars
from ml_tools.fixtures import articles
import numpy as np
import pandas as pd

lhs = articles()
lhs = Similars(lhs).embed().cluster(algo='agglomorative').value()

rhs = np.load('/storage/libgen/testing.npy') #, mmap_mode='r')
df = pd.read_feather('/storage/libgen/testing.df')
dnn = CosineEstimator(rhs)
adjustments = np.zeros((rhs.shape[0],))
# TODO hyperopt adjustments
dnn.hyperopt(lhs, adjustments, df, r"(cbt|virtual|cognitive)")
