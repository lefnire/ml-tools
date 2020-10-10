from ml_tools import CosineEstimator, Similars
from ml_tools.fixtures import articles
import numpy as np

corpus = articles()

split_ = len(corpus)//3
x, y = corpus[split_:], corpus[:split_]  # note reversal; x should be smaller
x, y = Similars(x, y).embed().value()

def test_cosine_estimator():
    dnn = CosineEstimator(y)
    dnn.fit_cosine()
    adjustments = np.zeros((y.shape[0],))
    dnn.fit_adjustments(x, adjustments)
    preds = dnn.predict(x)
    print(preds)
    # TODO test outcome. Will need a larger corpus with dissimilar articles
