from src.components.model_registery_deployment import register_the_results
from src.training_config import MODELS_TO_BE_TRAINED, PARAMETER_GRID, TrainConfiguration
from src.pipelines.training_pipeline import ModelTrainingPipeline
from sklearn.pipeline import make_pipeline
# register from training
from src.training_config import (MODELS_TO_BE_TRAINED, PARAMETER_GRID,
                                     TrainConfiguration,RawDataPathConfig)
from src.components.data_ingestion import GettDataIngestFromLocal
import pandas as pd
from src.utils import load_object
from src.logger import logging
from src.exceptions import CustomException
import sys

def register_models(models_dictionary):
    """
    Function to register models in mlflow
    args:
    models_dictionary:  a dictionary that has the data to be registered:
        'best_params':optimizer.best_params,
        'best_model' : best_model,
        'transformer' : original_transformer,
        'optimizer': optimizer.grid_search                                
    """

    # Ingesting data  to predict model performance to be registered:
    raw_data_path = RawDataPathConfig.raw_data_path 
    data_ingester = GettDataIngestFromLocal(path=raw_data_path)
    ingested_data_conf = data_ingester.ingest_data() # training_set_features
                                                     # test_set_features
                                                     # training_set_labels
                                                     # submission_format
    # Reading Data and creating data frames :
    X_training = pd.read_csv(ingested_data_conf.training_set_features)
    y_training = pd.read_csv(ingested_data_conf.training_set_labels)
    y_h1n1 = y_training['h1n1_vaccine']
    y_seasonal = y_training['seasonal_vaccine']

    try:
        for key in models_dictionary.keys():
            logging.info(f"registrgin model {key}")
            X_features = X_training.copy()
            print(key)
            _ ,label = key.split('_')[0],key.split('_')[1],
            best_params = models_dictionary[key]['best_params'] 
            best_model = models_dictionary[key]['best_model']
            preprocessor = models_dictionary[key]['transformer']

            final_model = make_pipeline(preprocessor,best_model)

            if label=='h1n1':
                y_train = y_h1n1.copy()
            else:
                y_train = y_seasonal.copy()

            final_model.fit(X_features,y_train)

            register_the_results(X_features,y_train,final_model,experiment_name='model_registry',run_name=f'{key}'
                                ,if_fit=False,run_params=best_params,tag=f'{key}',suffix=1,if_register=False,
                                model=final_model)
    except Exception as e:
        CustomException(e,sys)
        
        
def register_from_training_pipeline(cv_nfold=5, iter=10, scoring='roc_auc',save_models=True):
    """
    A helper function to tain model using training pipeline and register it in mlflow
    """
    logging.info('Registering models from training:')
    try:
        models_dictionart = ModelTrainingPipeline(MODELS_TO_BE_TRAINED,PARAMETER_GRID,
                                                    raw_data_config=RawDataPathConfig,
                                                    save_path_config=TrainConfiguration,save_models=save_models,
                                                    cv_nfold=cv_nfold, iter=iter, scoring=scoring)
        register_models(models_dictionart)
    except Exception as e:
        CustomException(e,sys)

def register_from_saved_models():
    """
    A helper function to load saved models and register it in mlflow
    TODO : not implemented
    """
    logging.info('Registering models from saved models:')
    try:
        models_data = load_object('file_path')
        models_dictionart = 'MODELS_TO_BE_TRAINED Load models from saved directory'
        register_models(models_dictionart)
    except Exception as e:
        CustomException("NotImplemented",sys)