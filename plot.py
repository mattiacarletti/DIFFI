import numpy as np
import pandas as pd
import os 
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.pyplot import cm
from matplotlib.pyplot import *
from utils import *
from tqdm import tqdm
import pickle
from interpretability_module import *

def compute_local_importances(model, X: pd.DataFrame,name: str,pwd_imp_score: str = os.getcwd(), pwd_plt_data: str = os.getcwd()) -> tuple[np.array,dict,str,str]:
    """
    Collect useful information that will be successively used by the plt_importances_bars,plt_global_importance_bar and plt_feat_bar_plot
    functions. 
    
    Parameters:
        model: An instance of the Isolation Forest model
        X: Input dataset   
        name: Dataset's name   
        pwd_imp_score: Directory where the Importance Scores results will be saved as pkl files, by default the current working directory
        pwd_plt_data: Directory where the plot data results will be saved as pkl files, by default the current working directory        
    
    Returns:
        imps: 2-dimensional array containing the local Feature Importance values for the samples of the input dataset X. The array is also locally saved in a pkl file for the sake of reproducibility.
        plt_data: Dictionary containig the average Importance Scores values, the feature order and the standard deviations on the Importance Scores. The dictionary is also locally saved in a pkl file for the sake of reproducibility.
        path_fi: Path of the pkl file containing the Importance Scores
        path_plt_data: Path of the pkl file containing the plt data        
    """

    name='LFI_'+name
    fi,_,_=local_diffi_batch(model,X)

    # Save the Importance Scores in a pkl file
    path_fi = pwd_imp_score + '\\imp_score_' + name + '.pkl'
    with open(path_fi, 'wb') as fl:
        pickle.dump(fi,fl)

    """ 
    Take the mean feature importance scores over the different runs for the Feature Importance Plot
    and put it in decreasing order of importance.
    To remove the possible np.nan or np.inf values from the mean computation use assign np.nan to the np.inf values 
    and then ignore the np.nan values using np.nanmean
    """

    fi[fi==np.inf]=np.nan
    mean_imp=np.nanmean(fi,axis=0)
    std_imp=np.nanstd(fi,axis=0)
    mean_imp_val=np.sort(mean_imp)
    feat_order=mean_imp.argsort()

    plt_data={'Importances': mean_imp_val,
              'feat_order': feat_order,
              'std': std_imp[mean_imp.argsort()]}
    
    # Save the plt_data dictionary in a pkl file
    path_plt_data = pwd_plt_data + '\\plt_data_' + name + '.pkl'
    with open(path_plt_data, 'wb') as fl:
        pickle.dump(plt_data,fl)
    

    return fi,plt_data,path_fi,path_plt_data

def compute_global_importances(model, X: pd.DataFrame, n_runs:int, name: str,pwd_imp_score: str = os.getcwd(), pwd_plt_data: str = os.getcwd()) -> tuple[np.array,dict,str,str]:
    """
    Collect useful information that will be successively used by the plt_importances_bars,plt_global_importance_bar and plt_feat_bar_plot
    functions. 
    
    Parameters:
        model: An instance of the Isolation Forest model
        X: Input Dataset
        n_runs: Number of runs to perform in order to compute the Global Feature Importance Scores.
        name: Dataset's name   
        pwd_imp_score: Directory where the Importance Scores results will be saved as pkl files, by default the current working directory
        pwd_plt_data: Directory where the plot data results will be saved as pkl files, by default the current working directory        
                
    Returns:
        imps: 2-dimensional array containing the local Feature Importance values for the samples of the input dataset X. The array is also locally saved in a pkl file for the sake of reproducibility.
        plt_data: Dictionary containig the average Importance Scores values, the feature order and the standard deviations on the Importance Scores. The dictionary is also locally saved in a pkl file for the sake of reproducibility.
        path_fi: Path of the pkl file containing the Importance Scores
        path_plt_data: Path of the pkl file containing the plt data    
    """

    name='GFI_'+name
    fi=np.zeros(shape=(n_runs,X.shape[1]))
    for i in range(n_runs):
        model.fit(X)
        fi[i,:],_=diffi_ib(model,X)

    # Save the Importance Scores in a pkl file
    path_fi = pwd_imp_score + '\\imp_score_' + name + '.pkl'
    with open(path_fi, 'wb') as fl:
        pickle.dump(fi,fl)
        

    fi[fi==np.inf]=np.nan
    mean_imp=np.nanmean(fi,axis=0)
    std_imp=np.nanstd(fi,axis=0)
    mean_imp_val=np.sort(mean_imp)
    feat_order=mean_imp.argsort()

    plt_data={'Importances': mean_imp_val,
              'feat_order': feat_order,
              'std': std_imp[mean_imp.argsort()]}
    
    # Save the plt_data dictionary in a pkl file
    path_plt_data = pwd_plt_data + '\\plt_data_' + name + '.pkl'
    with open(path_plt_data, 'wb') as fl:
        pickle.dump(plt_data,fl)
    

    return fi,plt_data,path_fi,path_plt_data

def plt_importances_bars(imps_path: str, name: str, pwd: str =os.getcwd(),f: int = 6,save: bool =True):
    """
    Obtain the Global Importance Bar Plot given the Importance Scores values computed in the compute_imps function. 
    
    Parameters:
        imps_path: Path of the pkl file containing the 2-dimensional array of the LFI/GFI Scores for the input dataset.Obtained from the compute_imps function.   
        name: Dataset's name 
        pwd: Directory where the results will be saved as pkl files. By default the value of pwd is set to the current working directory.    
        f: Number of vertical bars to include in the Bar Plot. By default f is set to 6.   
        save: Boolean variable used to decide weather to save the Bar Plot locally as a PDF or not.
    
    Returns:
        Obtain the Bar Plot which is then saved locally as a PDF.    
    """
    
    #Load the imps array from the pkl file contained in imps_path -> the imps_path is returned from the 
    #compute_local_importances or compute_global_importances functions so we have it for free 
    with open(imps_path, 'rb') as file:
        importances = pickle.load(file)

    number_colours = 20
    color = plt.cm.get_cmap('tab20',number_colours).colors
    patterns = [None, "/" , "\\" , "|" , "-" , "+" , "x", "o", "O", ".", "*" ]
    importances_matrix = np.array([np.array(pd.Series(x).sort_values(ascending = False).index).T for x in importances])
    dim=importances.shape[1]
    dim=int(dim)
    bars = [[(list(importances_matrix[:,j]).count(i)/len(importances_matrix))*100 for i in range(dim)] for j in range(dim)]
    bars = pd.DataFrame(bars)
    #display(bars)

    tick_names=[]
    for i in range(1,f+1):
        if i==1:
            tick_names.append(r'${}'.format(i) + r'^{st}$')
        elif i==2:
            tick_names.append(r'${}'.format(i) + r'^{nd}$')
        elif i==3:
            tick_names.append(r'${}'.format(i) + r'^{rd}$')
        else:
            tick_names.append(r'${}'.format(i) + r'^{th}$')

    barWidth = 0.85
    r = range(dim)
    ncols=1
    if importances.shape[1]>15:
        ncols=2


    fig, ax = plt.subplots()

    for i in range(dim):
        ax.bar(r[:f], bars.T.iloc[i, :f].values, bottom=bars.T.iloc[:i, :f].sum().values, color=color[i % number_colours], edgecolor='white', width=barWidth, label=str(i), hatch=patterns[i // number_colours])

    ax.set_xlabel("Rank", fontsize=20)
    ax.set_xticks(range(f), tick_names[:f])
    ax.set_ylabel("Percentage count", fontsize=20)
    ax.set_yticks(range(10, 101, 10), [str(x) + "%" for x in range(10, 101, 10)])
    ax.legend(bbox_to_anchor=(1.05, 0.95), loc="upper left",ncol=ncols)

    if save:
        plt.savefig(pwd + '//{}_bar_plot.pdf'.format(name), bbox_inches='tight')

    return fig, ax, bars


def plt_feat_bar_plot(plt_data_path: str,name: str,pwd: str =os.getcwd(),is_local: bool =True,save: bool =True):
    """
    Obtain the Global Feature Importance Score Plot exploiting the information obtained from compute_imps function. 
    
    Parameters
    ----------
        plt_data_path: Dictionary generated from the compute_imps function with the necessary information to create the Score Plot.
        name: Dataset's name
        pwd: Directory where the plot will be saved as pkl files. By default the value of pwd is set to the current working directory.  
        is_local: Boolean variable used to specify weather we are plotting the Global or Local Feature Importance in order to set the file name.
            If is_local is True the result will be the LFI Score Plot (based on the LFI scores of the input samples), otherwise the result is the GFI 
            Score Plot (based on the GFI scores obtained in the different n_runs execution of the model).
        save: Boolean variable used to decide weather to save the Score Plot locally as a PDF or not. 
                
    Returns:
        Obtain the Score Plot which is also locally saved as a PDF. 
    """
    #Load the plt_data dictionary from the pkl file contained in plt_data_path -> the plt_data_path is returned from the 
    #compute_local_importances or compute_global_importances functions so we have it for free 
    with open(plt_data_path, 'rb') as f:
        plt_data = pickle.load(f)

    name_file='Score_plot_'+name 

    patterns = [None, "/" , "\\" , "|" , "-" , "+" , "x", "o", "O", ".", "*" ]
    imp_vals=plt_data['Importances']
    feat_imp=pd.DataFrame({'Global Importance': np.round(imp_vals,3),
                          'Feature': plt_data['feat_order'],
                          'std': plt_data['std']
                          })
    
    if len(feat_imp)>15:
        feat_imp=feat_imp.iloc[-15:].reset_index(drop=True)
    
    dim=feat_imp.shape[0]

    number_colours = 20

    plt.style.use('default')
    plt.rcParams['axes.facecolor'] = '#F2F2F2'
    plt.rcParams['axes.axisbelow'] = True
    color = plt.cm.get_cmap('tab20',number_colours).colors
    ax1=feat_imp.plot(y='Global Importance',x='Feature',kind="barh",color=color[feat_imp['Feature']%number_colours],xerr='std',
                     capsize=5, alpha=1,legend=False,
                     hatch=[patterns[i//number_colours] for i in feat_imp['Feature']])
    xlim=np.min(imp_vals)-0.2*np.min(imp_vals)

    ax1.grid(alpha=0.7)
    ax2 = ax1.twinx()
    # Add labels on the right side of the bars
    values=[]
    for i, v in enumerate(feat_imp['Global Importance']):
        values.append(str(v) + ' +- ' + str(np.round(feat_imp['std'][i],2)))
    
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_yticks(range(dim))
    ax2.set_yticklabels(values)
    ax2.grid(alpha=0)
    plt.axvline(x=0, color=".5")
    ax1.set_xlabel('Importance Score',fontsize=20)
    ax1.set_ylabel('Features',fontsize=20)
    plt.xlim(xlim)
    plt.subplots_adjust(left=0.3)
    if save:
        plt.savefig(pwd+'//{}.pdf'.format(name_file),bbox_inches='tight')
        
    return ax1,ax2


def plot_importance_map(name: str,model, X_train: pd.DataFrame,y_train: np.array ,resolution: int,
                        pwd: str =os.getcwd(),save: bool =True,m: bool =None,factor: int =3,feats_plot: tuple[int,int] =(0,1),ax=None):
    """
    Produce the Local Feature Importance Scoremap.   
    
    Parameters:
        name: Dataset's name
        model: Instance of the Isolation Forest model. 
        X_train: Training Set 
        y_train: Dataset training labels
        resolution: Scoremap resolution 
        pwd: Directory where the plot will be saved. By default the value of pwd is set to the current working directory.
        save: Boolean variable used to decide weather to save the Score Plot locally as a PDF or not.  
        m: Boolean variable regulating the plt.pcolor advanced settings. By defualt the value of m is set to None
        factor: Integer factor used to define the minimum and maximum value of the points used to create the scoremap. By default the value of f is set to 3.
        feats_plot: This tuple contains the indexes of the pair features to compare in the Scoremap. By default the value of feats_plot
                is set to (0,1)
        plt: Plt object used to create the plot.  
                 
    Returns:
        Obtain the Scoremap which is also locally saved as a PDF. 
    """
    mins = X_train.min(axis=0)[list(feats_plot)]
    maxs = X_train.max(axis=0)[list(feats_plot)]  
    mean = X_train.mean(axis = 0)
    mins = list(mins-(maxs-mins)*factor/10)
    maxs = list(maxs+(maxs-mins)*factor/10)
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution), np.linspace(mins[1], maxs[1], resolution))
    mean = np.repeat(np.expand_dims(mean,0),len(xx)**2,axis = 0)
    mean[:,feats_plot[0]]=xx.reshape(len(xx)**2)
    mean[:,feats_plot[1]]=yy.reshape(len(yy)**2)

    importance_matrix = np.zeros_like(mean)
    model.max_samples = len(X_train)
    for i in range(importance_matrix.shape[0]):
        importance_matrix[i] = local_diffi(model, mean[i])[0]
    
    sign = np.sign(importance_matrix[:,feats_plot[0]]-importance_matrix[:,feats_plot[1]])
    Score = sign*((sign>0)*importance_matrix[:,feats_plot[0]]+(sign<0)*importance_matrix[:,feats_plot[1]])
    x = X_train[:,feats_plot[0]].squeeze()
    y = X_train[:,feats_plot[1]].squeeze()
    
    Score = Score.reshape(xx.shape)

    # Create a new pyplot object if plt is not provided
    if ax is None:
        fig, ax = plt.subplots()
    
    if m is not None:
        cp = ax.pcolor(xx, yy, Score, cmap=cm.RdBu, vmin=-m, vmax=m, shading='nearest')
    else:
        cp = ax.pcolor(xx, yy, Score, cmap=cm.RdBu, shading='nearest', norm=colors.CenteredNorm())
    
    ax.contour(xx, yy, (importance_matrix[:, feats_plot[0]] + importance_matrix[:, feats_plot[1]]).reshape(xx.shape), levels=7, cmap=cm.Greys, alpha=0.7)

    try:
        ax.scatter(x[y_train == 0], y[y_train == 0], s=40, c="tab:blue", marker="o", edgecolors="k", label="inliers")
        ax.scatter(x[y_train == 1], y[y_train == 1], s=60, c="tab:orange", marker="*", edgecolors="k", label="outliers")
    except IndexError:
        print('Handling the IndexError Exception...')
        ax.scatter(x[(y_train == 0)[:, 0]], y[(y_train == 0)[:, 0]], s=40, c="tab:blue", marker="o", edgecolors="k", label="inliers")
        ax.scatter(x[(y_train == 1)[:, 0]], y[(y_train == 1)[:, 0]], s=60, c="tab:orange", marker="*", edgecolors="k", label="outliers")

    ax.legend()
    if save:
        plt.savefig(pwd + '\\Local_Importance_Scoremap_{}.pdf'.format(name), bbox_inches='tight')
    else: 
        fig,ax=None,None

    return fig, ax

def plot_complete_scoremap(name:str,dim:int,model,X: pd.DataFrame, y: np.array, pwd:str =os.getcwd()):
        """
        Produce the Complete Local Feature Importance Scoremap: a Scoremap for each pair of features in the input dataset.   
        
        Parameters:
                name: Dataset's name
                dim: Number of input features in the dataset
                model: Instance of the Isolation Forest model. 
                X: Input dataset 
                y: Dataset labels
                pwd: Directory where the plot will be saved. By default the value of pwd is set to the current working directory.
        
        Returns:
                Obtain the Complete Scoremap which is also locally saved as a PDF. 
        """
            
        fig, ax = plt.subplots(dim, dim, figsize=(50, 50))
        for i in range(dim):
          for j in range(i+1,dim):
                features = [i,j]
                # One of the successive two lines can be commented so that we obtain only one "half" of the 
                #matrix of plots to reduce a little bit the execution time. 
                _,_=plot_importance_map(name,model, X, y, 50, pwd, feats_plot = (features[0],features[1]), ax=ax[i,j],save=False)
                _,_=plot_importance_map(name,model, X, y, 50, pwd, feats_plot = (features[1],features[0]), ax=ax[j,i],save=False)
                #fig.suptitle("comparison between DIFFI and ExIFFI "+name+" dataset",fontsize=20)

        plt.savefig(pwd+'//Local_Importance_Scoremap_{}_complete.pdf'.format(name),bbox_inches='tight')
        return fig,ax


def print_score_map(model,X: pd.DataFrame,resolution: int ,name: str ,pwd: str =os.getcwd(),save: bool =True):
    """
    Produce the Anomaly Score Scoremap.   
    
    Parameters:
        model: Instance of the Isolation Forest model.
        X: Input dataset
        resolution: Scoremap resolution 
        name: Dataset's name
        pwd: Directory where the plot will be saved. By default the value of pwd is set to the current working directory.
        save: Boolean variable used to decide weather to save the Score Plot locally as a PDF or not.

    Returns:
        Returns the Anomaly Score Scoremap  
    """
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    mins = list(mins-(maxs-mins)*3/10)
    maxs = list(maxs+(maxs-mins)*3/10)
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution), np.linspace(mins[1], maxs[1], resolution))

    #S1 = model.Anomaly_Score(X_in=np.c_[xx.ravel(), yy.ravel()])
    S1=model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    S1 = S1.reshape(xx.shape)
    x= X.T[0]
    y= X.T[1]        

    plt.figure(figsize=(12,12)) 
    levels = np.linspace(np.min(S1),np.max(S1),10)
    CS = plt.contourf(xx, yy, S1, levels, cmap=plt.cm.YlOrRd)
    plt.scatter(x,y,s=15,c='None',edgecolor='k')
    plt.axis("equal")
    if save:
        plt.savefig(pwd+'\\Anomaly_Scoremap_{}.pdf'
                .format(name),bbox_inches='tight')
    plt.show()
    return