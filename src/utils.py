import pickle
import json
import os
from src.exceptions import CustomException
import sys
import numpy as np
from src.logger import logging
from sklearn.pipeline import make_pipeline
import pandas as pd

"""
Different helper function to save object, load object as well as evaluating different metrics in the models
"""

def save_object(file_path,filename ,obj):
    """_summary_

    Args:
        file_path (str): path to save the object
        filename (str): name of the file that the object will be saved as
        obj (any): abject to be saved as binary file

    """
    try:
        os.makedirs(file_path, exist_ok=True)
        save_file = os.path.join(file_path,filename)
        with open(save_file, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    """load object from path

    Args:
        file_path (str): path of the file to be loaded.

    Returns:
        obect loaded using pickel: _description_
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to lists
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def get_metrics(y_true, y_pred, y_pred_prob, suffix=1):
    """calculates metric from true labels, predicted labels and predicted probablities

    Args:
        y_true (array): true labels of the features that models is trained on
        y_pred (array): predicted labels
        y_pred_prob (array): predicted probablities
        suffix (int, optional): a suffix to be used in naming figures and values. Defaults to 1.

    Returns:
        dictionary:  {f'accuracy_{suffix}': round(acc, 2), f'precision_{suffix}': round(prec, 2),
            f'recall_{suffix}': round(recall, 2), f'entropy_{suffix}': round(entropy, 2)
           , f'roc_auc{suffix}': round(roc_score, 2)}
    """
    from sklearn.metrics import accuracy_score,precision_score,recall_score,log_loss,roc_auc_score
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    entropy = log_loss(y_true, y_pred_prob)
    roc_score = roc_auc_score(y_true,y_pred_prob)
    return {f'accuracy_{suffix}': round(acc, 2), f'precision_{suffix}': round(prec, 2),
            f'recall_{suffix}': round(recall, 2), f'entropy_{suffix}': round(entropy, 2)
           , f'roc_auc{suffix}': round(roc_score, 2)}

def get_metrics_from_class(model,X_train,y_train,suffix=1):
    """_summary_

    Args:
        model (ml model): _description_
        X_train (arary): _description_. features
        y_train (array): _description_. labels
        suffix (int, optional): _description_. Defaults to 1.

    Returns:
        dictionary: get_metrics(y_train, y_pred, y_pred_probablity, suffix=suffix)
    """
    y_pred = model.predict(X_train)
    try:
        y_pred_probablity = model.predict_proba(X_train)
        y_pred_probablity = y_pred_probablity[:,1]
    except:
        y_pred_probablity = model.decision_function(X_train)
        

    return get_metrics(y_train, y_pred, y_pred_probablity, suffix=suffix)

def create_roc_auc_plot(y_true,y_pred ,suffix=1,path_in='./'):
    """created a ROC_AUC for a classifiyer

    Args:
        y_true (array): _description_.training labels
        y_test (array): _description_. predicted values of the features
        suffix (str or int, optional): _description_. Defaults to 1.
        path_in (str, optional): _description_. Defaults to './'.

    Returns:
        str: _description_. path of the created ROC_AUC figure
    """
    import matplotlib.pyplot as plt
    from sklearn import metrics
    # metrics.RocCurveDisplay.from_estimator(clf, X_data, y_data) 
    metrics.RocCurveDisplay.from_predictions(y_true,y_pred)
    path_save = os.path.join(path_in,f'roc_auc_curve_{suffix}.png')
    plt.savefig(path_save)
    return path_save

def create_confusion_matrix_plot(y_true,y_test, suffix=1,path_in='./'):
    """created a confusion matrix for a classifiyer

    Args:
        y_true (array): _description_.training labels
        y_test (array): _description_. predicted values of the features
        suffix (str or int, optional): _description_. Defaults to 1.
        path_in (str, optional): _description_. Defaults to './'.

    Returns:
        str: _description_. path of the created confusion matrix figure
    """
    import matplotlib.pyplot as plt
    from sklearn import metrics
    metrics.ConfusionMatrixDisplay.from_predictions(y_true,y_test)
    path_save = os.path.join(path_in,f'confusion_matrix_{suffix}.png')
    plt.savefig(path_save)
    return path_save

def get_cross_val_AUC(clfiyer,X_train, Y_Train,cv=3):
    """
    a function that return a score of cross validation in a model
    """
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score
    scores = cross_val_predict(clfiyer,X_train, Y_Train,cv=cv,method='predict_proba')
    """TODO check scores"""
    return round(roc_auc_score(Y_Train,scores[:,1]),3)  

def save_model_data(save_path,optimizer,transformer,lbl_name):
    """a helper function to save a model and its transformer in a path

    Args:
        save_path (str): _description_
        optimizer (sklearn optimizer): _description_
        transformer (transformer): _description_
        lbl_name (str): _description_
    """
    os.makedirs(save_path,exist_ok=True)
    save_object(save_path,f'model_{lbl_name}' ,optimizer.best_model)
    save_object(save_path,f'optimizer_{lbl_name}' ,optimizer.grid_search)
    save_object(save_path,f'transformer_{lbl_name}' ,transformer)
    with open(os.path.join(save_path,f'best_params_{lbl_name}.json'), 'w') as json_file:
        json.dump(optimizer.best_params, json_file,default=convert_numpy_types, indent=4)

def get_best_predictor_pipelines(csv_path):
    """A function that returns pipelines from csv file that incluedes paths and metrics of the different 
    trained models

    Args:
        csv_path (str): _description_. Path for a csv files 

    Returns:
        model pipelines from preprocessor and models : _description_
    """
    model_metrics = pd.read_csv(csv_path)
    model_1_path = model_metrics['save_path'].loc[0]
    model_2_path = model_metrics['save_path'].loc[1]

    model_1 = load_object(os.path.join(model_1_path,'model_h1n1'))
    transformer_1 = load_object(os.path.join(model_1_path,'transformer_h1n1'))

    logging.info(f"The model 1 chosen is {type(model_1).__name__}")

    model_2 = load_object(os.path.join(model_2_path,'model_seasonal'))
    transformer_2 = load_object(os.path.join(model_2_path,'transformer_h1n1'))

    logging.info(f"The model 2 chosen is {type(model_2).__name__}")

    pipline_1 = make_pipeline(transformer_1,model_1)
    pipline_2 = make_pipeline(transformer_2,model_2)

    return pipline_1,pipline_2