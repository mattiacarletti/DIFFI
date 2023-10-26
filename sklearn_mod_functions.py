# New BSD License

# Copyright (c) 2007–2020 The scikit-learn developers.
# All rights reserved.
# 

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

#   a. Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#   b. Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#   c. Neither the name of the Scikit-learn Developers  nor the names of
#      its contributors may be used to endorse or promote products
#      derived from this software without specific prior written
#      permission. 
# 

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.



from sklearn.ensemble._iforest import _average_path_length
from sklearn.utils.validation import  _num_samples
from sklearn.utils import gen_batches, get_chunk_n_rows
import numpy as np 

# The functions below have been adapted from the sklearn source code

def decision_function_single_tree(iforest, tree_idx, X):
    return _score_samples(iforest, tree_idx, X) - iforest.offset_


def _score_samples(iforest, tree_idx, X):
    n_feat= X.shape[1]
    if n_feat != X.shape[1]:
        raise ValueError("Number of features of the model must "
                         "match the input. Model n_features is {0} and "
                         "input n_features is {1}."
                         "".format(n_feat, X.shape[1]))
    return -_compute_chunked_score_samples(iforest, tree_idx, X)


def _compute_chunked_score_samples(iforest, tree_idx, X):
    n_samples = _num_samples(X)
    if iforest._max_features == X.shape[1]:
        subsample_features = False
    else:
        subsample_features = True
    chunk_n_rows = get_chunk_n_rows(row_bytes=16 * iforest._max_features,
                                    max_n_rows=n_samples)
    slices = gen_batches(n_samples, chunk_n_rows)
    scores = np.zeros(n_samples, order="f")
    for sl in slices:
        scores[sl] = _compute_score_samples_single_tree(iforest, tree_idx, X[sl], subsample_features)
    return scores


def _compute_score_samples_single_tree(iforest, tree_idx, X, subsample_features):
    n_samples = X.shape[0]
    depths = np.zeros(n_samples, order="f")
    tree = iforest.estimators_[tree_idx]
    features = iforest.estimators_features_[tree_idx]
    X_subset = X[:, features] if subsample_features else X
    leaves_index = tree.apply(X_subset)
    node_indicator = tree.decision_path(X_subset)
    n_samples_leaf = tree.tree_.n_node_samples[leaves_index]
    depths += (np.ravel(node_indicator.sum(axis=1)) + _average_path_length(n_samples_leaf) - 1.0)
    scores = 2 ** (-depths / (1 * _average_path_length([iforest.max_samples_])))
    return scores
