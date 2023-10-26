import numpy as np
from sklearn.ensemble import IsolationForest
import os
import sys
#parent_dir = os.path.abspath(os.path.join(os.path.dirname('DIFFI'), '..'))
#sys.path.append(parent_dir)
from interpretability_module import *

def test_diffi_ib():
    # create a random dataset
    np.random.seed(0)
    X = np.random.randn(100, 10)
    # create an isolation forest model
    iforest = IsolationForest(n_estimators=10, random_state=0)
    iforest.fit(X)
    # run the diffi_ib function
    fi_ib, exec_time = diffi_ib(iforest, X)
    #Check that all the elements of fi_ib are finite
    assert np.all(np.isfinite(fi_ib)) == True 
    # check that the output has the correct shape
    assert fi_ib.shape == X.shape[1]
    # check that the execuiton time is positive
    assert exec_time > 0 

def test_get_iic():
    # create a random dataset
    np.random.seed(0)
    X = np.random.randn(100, 10)
    # create an isolation forest model
    iforest = IsolationForest(n_estimators=10, random_state=0)
    iforest.fit(X)
    estimator=iforest.estimators_[np.random.randint(0,iforest.n_estimators)]
    is_leaves=np.random.choice([True, False], size=X.shape[0])
    adjust_iic=np.random.choice([True, False], size=1)
    lambda_outliers_ib = _get_iic(estimator, X, is_leaves, adjust_iic=adjust_iic)

    assert type(lambda_outliers_ib) == np.ndarray
    assert lambda_outliers_ib.shape == estimator.tree_.node_count
    assert np.all(lambda_outliers_ib >= -1) == True 


def test_local_diffi():
    # create a random dataset
    np.random.seed(0)
    # local_diffi works on a single sample
    X = np.random.randn(10)
    # create an isolation forest model
    iforest = IsolationForest(n_estimators=10, random_state=0)
    iforest.fit(X)

    fi_ib, exec_time = local_diffi(iforest, X)

    assert np.all(np.isfinite(fi_ib)) == True
    assert fi_ib.shape == X.shape[0]
    assert exec_time > 0




