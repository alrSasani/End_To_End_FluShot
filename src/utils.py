import pickle
import json
import os
from src.exceptions import CustomException
import sys
import numpy as np


"""
Different helper function to save object, load object as well as evaluating different metrics in the models
"""

def save_object(file_path,filename ,obj):
    try:
        os.makedirs(file_path, exist_ok=True)
        save_file = os.path.join(file_path,filename)
        with open(save_file, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
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
    y_pred = model.predict(X_train)
    try:
        y_pred_probablity = model.predict_proba(X_train)
        y_pred_probablity = y_pred_probablity[:,1]
    except:
        y_pred_probablity = model.decision_function(X_train)
        

    return get_metrics(y_train, y_pred, y_pred_probablity, suffix=suffix)

def create_roc_auc_plot(y_true,y_pred ,suffix=1,path_in='./'):
    import matplotlib.pyplot as plt
    from sklearn import metrics
    # metrics.RocCurveDisplay.from_estimator(clf, X_data, y_data) 
    metrics.RocCurveDisplay.from_predictions(y_true,y_pred)
    path_save = os.path.join(path_in,f'roc_auc_curve_{suffix}.png')
    plt.savefig(path_save)
    return path_save

def create_confusion_matrix_plot(y_true,y_test, suffix=1,path_in='./'):
    import matplotlib.pyplot as plt
    from sklearn import metrics
    metrics.ConfusionMatrixDisplay.from_predictions(y_true,y_test)
    path_save = os.path.join(path_in,f'confusion_matrix_{suffix}.png')
    plt.savefig(path_save)
    return path_save

def get_cross_val_AUC(clfiyer,X_train, Y_Train,cv=3):
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score
    scores = cross_val_predict(clfiyer,X_train, Y_Train,cv=cv,method='predict_proba')
    """TODO check scores"""
    return round(roc_auc_score(Y_Train,scores[:,1]),3)  

def save_model_data(save_path,optimizer,transformer,lbl_name):
    os.makedirs(save_path,exist_ok=True)
    save_object(save_path,f'model_{lbl_name}' ,optimizer.best_model)
    save_object(save_path,f'optimizer_{lbl_name}' ,optimizer.grid_search)
    save_object(save_path,f'transformer_{lbl_name}' ,transformer)
    with open(os.path.join(save_path,f'best_params_{lbl_name}.json'), 'w') as json_file:
        json.dump(optimizer.best_params, json_file,default=convert_numpy_types, indent=4)


