import os 
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.pyplot import cm
from matplotlib.pyplot import *
from utils import *
from tqdm import tqdm
import pickle
from interpretability_module import *
from plot import * 

def test_compute_local_importances():
    
    #Create a path to save the pkl files created by compute_local_importances
    test_imp_score_path=os.path.join(os.getcwd(),'tests','test_imp_score_local')
    test_plt_data_path=os.path.join(os.getcwd(),'tests','test_plt_data_local')
    name='test_local'

    #If the folder do not exist create them:
    if not os.path.exists(test_imp_score_path):
        os.makedirs(test_imp_score_path)
    if not os.path.exists(test_plt_data_path):
        os.makedirs(test_plt_data_path)

    np.random.seed(0)
    X = np.random.randn(100, 10)
    # create an isolation forest model
    iforest = IsolationForest(n_estimators=10, max_samples=64, random_state=0)
    iforest.fit(X)

    fi,plt_data,path_fi,path_plt_data=compute_local_importances(iforest,X,name,pwd_imp_score=test_imp_score_path,pwd_plt_data=test_plt_data_path)

    """
    Tests on the pkl files
    """
    #Check that the returned path are strings
    assert type(path_fi) == str
    assert type(path_plt_data) == str
    #Check that the pkl files have been created
    assert os.path.exists(path_fi) == True
    assert os.path.exists(path_plt_data) == True
    #Check that the pkl files are not empty
    assert os.path.getsize(path_fi) > 0
    assert os.path.getsize(path_plt_data) > 0
    #Check that the pkl files can be loaded
    assert pickle.load(open(path_fi,'rb')) is not None
    assert pickle.load(open(path_plt_data,'rb')) is not None

    """
    Tests on fi and plt_data
    """
    #Check that all the elements of fi are finite
    assert np.all(np.isfinite(fi)) == True
    # check that the output has the correct shape
    assert fi.shape[0] == X.shape[0]
    #Extract the keys of plt_data
    plt_data_keys=list(plt_data.keys())
    imp,feat_ord,std=plt_data[plt_data_keys[0]],plt_data[plt_data_keys[1]],plt_data[plt_data_keys[2]]
    #Check that all the elements of imp are finite
    assert np.all(np.isfinite(imp)) == True
    #Check that the size of imp is correct
    assert imp.shape[0] == X.shape[1]
    #Check that the size of feat_ord is correct
    assert feat_ord.shape[0] == X.shape[1]
    #Values in feat_ord cannot be greater than X.shape[1]
    assert np.all(feat_ord>=X.shape[1]) == False
    #Check that the size of std is correct
    assert std.shape[0] == X.shape[1]
    #Check that all the elements of std are positive (standard deviation cannot be negative)
    assert np.all(std>=0) == True


def test_compute_global_importances():

    #Create a path to save the pkl files created by compute_local_importances
    test_imp_score_path=os.path.join(os.getcwd(),'tests','test_imp_score_global')
    test_plt_data_path=os.path.join(os.getcwd(),'tests','test_plt_data_global')
    name='test_global'

    #If the folder do not exist create them:
    if not os.path.exists(test_imp_score_path):
        os.makedirs(test_imp_score_path)
    if not os.path.exists(test_plt_data_path):
        os.makedirs(test_plt_data_path)

    np.random.seed(0)
    X = np.random.randn(100, 10)
    # create an isolation forest model
    iforest = IsolationForest(n_estimators=10, max_samples=64, random_state=0)
    iforest.fit(X)
    nruns=np.random.randint(1,10)

    fi,plt_data,path_fi,path_plt_data=compute_global_importances(iforest,X,nruns,name,pwd_imp_score=test_imp_score_path,pwd_plt_data=test_plt_data_path)

    """
    Tests on the pkl files
    """

    #Check that the returned path are strings
    assert type(path_fi) == str
    assert type(path_plt_data) == str
    #Check that the pkl files have been created
    assert os.path.exists(path_fi) == True
    assert os.path.exists(path_plt_data) == True
    #Check that the pkl files are not empty
    assert os.path.getsize(path_fi) > 0
    assert os.path.getsize(path_plt_data) > 0
    #Check that the pkl files can be loaded
    assert pickle.load(open(path_fi,'rb')) is not None
    assert pickle.load(open(path_plt_data,'rb')) is not None

    """
    Tests on fi and plt_data
    """
    #Check that nruns is positive 
    assert nruns >= 0
    #Check that all the elements of fi are finite
    assert np.all(np.isfinite(fi)) == True
    # check that the output has the correct shape
    assert fi.shape[1] == X.shape[1]
    #Extract the keys of plt_data
    plt_data_keys=list(plt_data.keys())
    imp,feat_ord,std=plt_data[plt_data_keys[0]],plt_data[plt_data_keys[1]],plt_data[plt_data_keys[2]]
    #Check that all the elements of imp are finite
    assert np.all(np.isfinite(imp)) == True
    #Check that the size of imp is correct
    assert imp.shape[0] == X.shape[1]
    #Check that the size of feat_ord is correct
    assert feat_ord.shape[0] == X.shape[1]
    #Values in feat_ord cannot be greater than X.shape[1]
    assert np.all(feat_ord>=X.shape[1]) == False
    #Check that the size of std is correct
    assert std.shape[0] == X.shape[1]
    #Check that all the elements of std are positive (standard deviation cannot be negative)
    assert np.all(std>=0) == True

def test_plot_importances_bars():
    
    assert True

def test_plt_feat_bar_plot():
    
    assert True

def test_plot_importance_map():
    
    assert True

def test_plot_complete_scoremap():
        
        assert True

