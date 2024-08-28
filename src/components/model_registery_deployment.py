import sys
import mlflow
from src.utils import (get_cross_val_AUC,get_metrics_from_class,get_metrics,
                       create_confusion_matrix_plot,create_roc_auc_plot)
from src.logger import logging
from src.exceptions import CustomException

def create_exp_and_register_model(experiment_name,run_name,run_metrics, confusion_matrix_path_1 = None, 
                                   roc_auc_plot_path_1 = None, run_params=None,tag = 'SGD',if_register=False
                                  ,model=None):
    if if_register:
        mlflow.set_tracking_uri("http://localhost:5000") 
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        if  run_params is not None:
            for param in run_params:
                mlflow.log_param(param, run_params[param])  
        for metric in run_metrics:
            mlflow.log_metric(metric, run_metrics[metric])        
        if confusion_matrix_path_1 is not None:
            mlflow.log_artifact(confusion_matrix_path_1, 'confusion_materix_1')
        if roc_auc_plot_path_1 is not None:
            mlflow.log_artifact(roc_auc_plot_path_1, "roc_auc_plot_1")            
        if model is not None:    
            mlflow.sklearn.log_model(model, "model",registered_model_name=f"{experiment_name}_{run_name}")        
        mlflow.set_tag("tag1", tag)


def register_the_results(X_train,y_train,model_func,experiment_name='Experiment',run_name='run'
                         ,if_fit=True,run_params={},tag='RandomForest',suffix=1,if_register=False,
                         model=None):
    try:
        logging.info(f"Registering the model {type(model_func).__name__} in mlflow")
        if if_fit:
            model = model_func(X_train,y_train,**run_params)
        else:
            model = model_func
        
        y_pred = model.predict(X_train)
        try:
            y_pred_probablity = model.predict_proba(X_train)
            y_pred_probablity = y_pred_probablity[:,1]
        except:
            y_pred_probablity = model.decision_function(X_train)
            

        run_metrics =get_metrics(y_train, y_pred, y_pred_probablity, suffix=suffix)

        cv_roc_score = get_cross_val_AUC(model,X_train,y_train)
        run_metrics['cv_roc_score'] = cv_roc_score

        path_con_mx = create_confusion_matrix_plot( y_train,y_pred, suffix=suffix)
        
        path_auc = create_roc_auc_plot(y_train, y_pred_probablity, suffix=suffix)
        
        create_exp_and_register_model(experiment_name,run_name,run_metrics, confusion_matrix_path_1 = path_con_mx, 
                                    roc_auc_plot_path_1 = path_auc,run_params=run_params, tag = tag,
                                    if_register=if_register, model=model)  
    except Exception as e:
        CustomException(e,sys)  
    
