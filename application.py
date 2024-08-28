import os
import sys
from pathlib import Path
from model_deployment.manage import main  # Import the main function from manage.py
from src.pipelines import training_pipeline,mlflow_model_registry_pipeline

from src.training_config import (MODELS_TO_BE_TRAINED, PARAMETER_GRID,
                                     TrainConfiguration,RawDataPathConfig)

def application():
    # main_path = os.getcwd()

    mlflow_model_registry_pipeline.register_from_training_pipeline(cv_nfold=2, iter=2, scoring='roc_auc',
                                                                   save_models=True)
    # training_pipeline.ModelTrainingPipeline(MODELS_TO_BE_TRAINED,PARAMETER_GRID,raw_data_config=RawDataPathConfig,save_path_config=TrainConfiguration,
    #                       save_models=True,cv_nfold=2, iter=2, scoring='roc_auc')
    
    os.system('python model_deployment/manage.py runserver')
    
if __name__ == "__main__":
    os.system('python setup.py install')
    application()
