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

    # We need a feature importance 2d array with the importance values. 
    # We can extract it from the pkl files created by the test_compure_global_importances 
    # and test_compute_local_importances functions

    #We create the plot with plot_importances_bars and we will then compare it with the 
    #expected result contained in GFI_glass_synt.pdf
    imps_path=os.path.join(os.getcwd(),'imp_scores','imp_score_GFI_glass.pkl')

    imps=pickle.load(open(imps_path,'rb'))

    #Create a path to save the plot image 
    plot_path=os.path.join(os.getcwd(),'tests','test_plots')

    #If the folder do not exist create it:
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    #Create a name for the plot
    name='test_Glass'
    f=6
    fig,ax,bars=plt_importances_bars(imps_path,name,pwd=plot_path,f=f)

    """
    Tests on ax
    """
    #Check that the returned ax is not None
    assert ax is not None
    assert fig is not None
    #Check that the returned ax is an axis object
    #assert type(ax) == matplotlib.axes._subplots.AxesSubplot
    #Check that the x label is correct
    assert ax.get_xlabel() == 'Rank'
    #Check that the y label is correct
    assert ax.get_ylabel() == 'Percentage count'
    #Check that the xtick  and y tick labels are correct
    x_tick_labels = [tick.get_text() for tick in ax.get_xticklabels()]
    y_tick_labels = [tick.get_text() for tick in ax.get_yticklabels()]
    assert x_tick_labels == ['$1^{st}$', '$2^{nd}$', '$3^{rd}$', '$4^{th}$', '$5^{th}$', '$6^{th}$']
    assert y_tick_labels == ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%']

    #See if the plot correctly changes if I pass from f=6 (default value) to f=9
    f1=9
    fig1,ax1,bars1=plt_importances_bars(imps_path,name='test_Glass_9',pwd=plot_path,f=f1)

    #Check that the xtick  and y tick labels are correct
    x_tick_labels1 = [tick.get_text() for tick in ax1.get_xticklabels()]
    assert x_tick_labels1 == ['$1^{st}$', '$2^{nd}$', '$3^{rd}$', '$4^{th}$', '$5^{th}$', '$6^{th}$','$7^{th}$','$8^{th}$','$9^{th}$']

    """
    Tests on bars

    The main test o perform on bars is that the sum of the percentages values on each column should be 100. 
    """
    assert type(bars) == pd.DataFrame
    assert bars.shape == (imps.shape[1],imps.shape[1])
    assert np.all(bars.sum()==100) == True
    #Same on bars1
    assert type(bars1) == pd.DataFrame
    assert bars1.shape == (imps.shape[1],imps.shape[1])
    assert np.all(bars1.sum()==100) == True

    """
    Comaparison with the expected image. Convert the images in png and compare
    their pixel values using the Pillow library -> see ChatGPT. 
    """

def test_plt_feat_bar_plot():

    # We need the plt_data array: let's consider the global case with plt_data_GFI_glass.pkl and 
    # the local case with plt_data_LFI_glass.pkl

    plt_data_global_path=os.path.join(os.getcwd(),'plt_data','plt_data_GFI_glass.pkl')
    plt_data_local_path=os.path.join(os.getcwd(),'plt_data','plt_data_LFI_glass.pkl')

    name_global='test_GFI_Glass'
    name_local='test_LFI_Glass'

    plot_path=os.path.join(os.getcwd(),'tests','test_plots')

    ax1,ax2=plt_feat_bar_plot(plt_data_global_path,name_global,pwd=plot_path,is_local=False)
    ax3,ax4=plt_feat_bar_plot(plt_data_local_path,name_local,pwd=plot_path,is_local=True)

    y_tick_labels_local = [tick.get_text() for tick in ax3.get_yticklabels()]
    y_tick_labels2_local = [tick.get_text() for tick in ax4.get_yticklabels()]
    y_tick_labels_global = [tick.get_text() for tick in ax1.get_yticklabels()]
    y_tick_labels2_global = [tick.get_text() for tick in ax2.get_yticklabels()]

    """
    Tests on ax1,ax2,ax3,ax4
    """
    #Check that the returned ax is not None
    assert ax1 is not None
    assert ax2 is not None
    assert ax3 is not None
    assert ax4 is not None
    #Check that the x label is correct
    assert ax1.get_xlabel() == 'Importance Score'
    #Check that the y label is correct
    assert ax1.get_ylabel() == 'Features'
     #Check that the x label is correct
    assert ax3.get_xlabel() == 'Importance Score'
    #Check that the y label is correct
    assert ax3.get_ylabel() == 'Features'
    #Check that the xtick  and y tick labels are correct
    assert np.all(np.array(y_tick_labels_local).astype('float')>=len(y_tick_labels2_local)-1) == False
    assert np.all(np.array(y_tick_labels_global).astype('float')>=len(y_tick_labels2_global)-1) == False
    
    assert True

def test_plot_importance_map():

    # Let's perform the test on the Glass dataset 
    with open(os.path.join(os.getcwd(), 'data', 'local', 'glass.pkl'), 'rb') as f:
        data = pickle.load(f)
    # training data (inliers and outliers)
    X_tr = np.concatenate((data['X_in'], data['X_out_5'], data['X_out_6']))
    y_tr = np.concatenate((data['y_in'], data['y_out_5'], data['y_out_6']))
    X_tr, y_tr = shuffle(X_tr, y_tr, random_state=0)
    # test outliers
    X_te = data['X_out_7'] 
    y_te = data['y_out_7']
    y_te=np.ones(shape=X_te.shape[0])
    X=np.r_[X_tr,X_te]
    y=np.r_[y_tr,y_te]    
    name='Glass'
    # create an isolation forest model
    iforest = IsolationForest(n_estimators=10, max_samples=64, random_state=0)
    iforest.fit(X_tr)
    plot_path=os.path.join(os.getcwd(),'tests','test_plots')

    fig,ax=plot_importance_map(name,iforest,X,y,30,pwd=plot_path)

    """
    Tests on ax
    """

    #Check that the returned ax is not None
    assert ax is not None
    assert fig is not None

def test_plot_complete_scoremap():
        
    # Here we'll use a random dataset with just 3 features otherwise it takes too much time to 
    #create the plots
    np.random.seed(0)
    X = np.random.randn(100, 3)
    #Assign at random the anomalous/not anomaoous labels
    #Create a random array of 0 and 1 of shape=(100,)
    y=np.random.randint(0,2,size=100)
    name='test_complete'
    # create an isolation forest model
    iforest = IsolationForest(n_estimators=10, max_samples=64, random_state=0)
    iforest.fit(X)
    plot_path=os.path.join(os.getcwd(),'tests','test_plots')

    fig,ax=plot_complete_scoremap(name,X.shape[1],iforest,X,y,pwd=plot_path)
        
    """
    Tests on ax
    """

    #Check that the returned ax is not None
    assert ax is not None
    assert fig is not None

