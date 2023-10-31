from sklearn.ensemble._iforest import _average_path_length
from sklearn.utils.validation import  _num_samples
from sklearn.utils import gen_batches, get_chunk_n_rows
from sklearn.ensemble import IsolationForest
import numpy as np 
from sklearn_mod_functions import * 
from sklearn.utils._testing import (
    assert_allclose,
    assert_array_almost_equal,
    assert_array_equal,
    ignore_warnings,
)

from sklearn_mod_functions import _score_samples
from sklearn_mod_functions import _compute_chunked_score_samples
from sklearn_mod_functions import _compute_score_samples_single_tree

def test_decision_function_single_tree():

    X_train = np.array([[1, 1], [1, 2], [2, 1]])
    clf1 = IsolationForest(contamination=0.1).fit(X_train)
    clf2 = IsolationForest().fit(X_train)
    X=np.array([[2.0, 2.0]])
    tree_idx=np.random.randint(0,len(clf1.estimators_))

    assert_array_equal(
        decision_function_single_tree(clf1,tree_idx,X),
        _score_samples(clf1,tree_idx,X) - clf1.offset_,
    )
    assert_array_equal(
        decision_function_single_tree(clf2,tree_idx,X),
        _score_samples(clf2,tree_idx,X) - clf2.offset_,
    )

    #The decision function values could not be equal because clf1 and clf2 have 
    #two different contamination values. 

    assert_array_almost_equal(
        decision_function_single_tree(clf1,tree_idx,X), decision_function_single_tree(clf2,tree_idx,X),
        decimal=1
    )

    #Check weather the two decision function values are different

    assert not np.array_equal(decision_function_single_tree(clf1,tree_idx,X), decision_function_single_tree(clf2,tree_idx,X))

def test_score_samples():

    X_train = np.array([[1, 1], [1, 2], [2, 1]])
    clf1 = IsolationForest(contamination=0.1).fit(X_train)
    clf2 = IsolationForest().fit(X_train)
    X=np.array([[2.0, 2.0]])
    tree_idx=np.random.randint(0,len(clf1.estimators_))
    assert_array_equal(
        _score_samples(clf1,tree_idx,X),
        decision_function_single_tree(clf1,tree_idx,X) + clf1.offset_,
    )
    assert_array_equal(
        _score_samples(clf2,tree_idx,X),
        decision_function_single_tree(clf2,tree_idx,X) + clf2.offset_,
    )
    assert_array_equal(
        _score_samples(clf1,tree_idx,X), _score_samples(clf2,tree_idx,X)
    )

def test_compute_chunked_score_samples():

    X_train = np.array([[1, 1], [1, 2], [2, 1]])
    clf1 = IsolationForest(contamination=0.1).fit(X_train)
    clf2 = IsolationForest().fit(X_train)
    X=np.array([[2.0, 2.0]])
    tree_idx=np.random.randint(0,len(clf1.estimators_))

    assert not np.array_equal(
        _compute_chunked_score_samples(clf1,tree_idx,X),
        decision_function_single_tree(clf1,tree_idx,X) + clf1.offset_,
    )
    assert not np.array_equal(
        _compute_chunked_score_samples(clf2,tree_idx,X),
        decision_function_single_tree(clf2,tree_idx,X) + clf2.offset_,
    )
    
    assert_array_equal(
        _compute_chunked_score_samples(clf1,tree_idx,X), _compute_chunked_score_samples(clf2,tree_idx,X)
    )

def test_compute_score_samples_single_tree():

    X_train = np.array([[1, 1], [1, 2], [2, 1]])
    clf1 = IsolationForest(contamination=0.1).fit(X_train)
    clf2 = IsolationForest().fit(X_train)
    X=np.array([[2.0, 2.0]])
    tree_idx=np.random.randint(0,len(clf1.estimators_))
    subsample_features=np.random.choice([True, False], size=1)

    assert not np.array_equal(
        _compute_score_samples_single_tree(clf1,tree_idx,X,subsample_features),
        decision_function_single_tree(clf1,tree_idx,X) + clf1.offset_,
    )
    assert not np.array_equal(
        _compute_score_samples_single_tree(clf2,tree_idx,X,subsample_features),
        decision_function_single_tree(clf2,tree_idx,X) + clf2.offset_,
    )

    assert_array_equal(
        _compute_score_samples_single_tree(clf1,tree_idx,X,subsample_features), _compute_score_samples_single_tree(clf2,tree_idx,X,subsample_features)
    )


