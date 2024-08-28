import numpy as np
import sys
import os
import pandas as pd
from catboost import CatBoostClassifier
import json
from src.components.data_ingestion import GettDataIngestFromLocal  
from src.components.data_transformers import (GetCatboostTransformer,
                                              GetSimpleTransformer,
                                              GetDefaultsPreprocessor,
                                              SimpleNullFiller)
from src.components.model_optimizers import RandomizeSearchModelOptimizer, GridSearchModelOptimizer
from src.utils import  get_cross_val_AUC, save_model_data, get_metrics_from_class
from src.logger import logging
from src.exceptions import CustomException


def ModelTrainingPipeline(model_to_be_trained,Prameters_gris,raw_data_config,
                          save_path_config, save_models=False,cv_nfold=5, iter=10, scoring='roc_auc'):
    try:
        logging.info("Training and optimizing the models...")
        # # data ingestion:
        raw_data_path = raw_data_config.raw_data_path 
        data_ingester = GettDataIngestFromLocal(path=raw_data_path)
        logging.info(f"Reading data from {raw_data_path}")
        ingested_data_conf = data_ingester.ingest_data() # training_set_features
                                                        # test_set_features
                                                        # training_set_labels
                                                        # submission_format
        # Reading Data :
        X_training = pd.read_csv(ingested_data_conf.training_set_features)
        y_training = pd.read_csv(ingested_data_conf.training_set_labels)
        y_h1n1 = y_training['h1n1_vaccine']
        y_seasonal = y_training['seasonal_vaccine']

        # # data_transformation:
        catb_trasformer = GetCatboostTransformer().get_transformer()
        features_cat_boost = catb_trasformer.fit_transform(X_training.copy())
        cat_feat_df = pd.DataFrame(features_cat_boost,columns=catb_trasformer.get_feature_names_out(),
                                index=X_training.index)
        
        cat_index = cat_feat_df.select_dtypes(include=object).columns
        default_transformer = GetDefaultsPreprocessor().get_transformer()
        features_rest = default_transformer.fit_transform(X_training.copy())

        best_models_h1n1 = {}
        best_models_seas = {}
        label_names  = ['h1n1','seasonal']

        metrics_df = pd.DataFrame()
        for model_name,model in model_to_be_trained.items():
            features = features_rest.copy()
            transformer = default_transformer
            original_transformer = GetDefaultsPreprocessor().get_transformer()
            if model_name == 'cat_boost':
                features = features_cat_boost.copy()
                model = CatBoostClassifier(cat_features=cat_index,verbose=0)
                transformer = catb_trasformer
                original_transformer = GetCatboostTransformer().get_transformer()

            for resu_dict,label,lbl_name in zip([best_models_h1n1,best_models_seas],[y_h1n1,y_seasonal],label_names):
                logging.info(f"Optimizing model {model_name} for label {lbl_name}")

                optimizer = RandomizeSearchModelOptimizer(model, Prameters_gris[model_name], 
                                                        cv_nfold=cv_nfold, iter=iter, scoring=scoring,
                                                        random_state=42)
                optimizer.optimize(features,label)
                best_model = optimizer.best_model.fit(features,label)
                
                metrics_model = get_metrics_from_class(best_model,features,label)
                cv_roc_score = get_cross_val_AUC(best_model,features,label)
                metrics_model['cv_roc_score'] = cv_roc_score
                temp_df = pd.DataFrame([metrics_model], index=[f'{model_name}_{lbl_name}'])

                resu_dict[f'{model_name}_{lbl_name}'] = {
                                        'best_params':optimizer.best_params,
                                        'best_model' : best_model,
                                        'transformer' : original_transformer,
                                        'optimizer': optimizer.grid_search
                                        }
                if save_models:
                    save_path = os.path.join(save_path_config.save_artifact_path,'Models_data',model_name)
                    logging.info(f"Saveing model {type(best_model).__name__} data to {save_path}")
                    save_model_data(save_path,optimizer,transformer,lbl_name)
                    temp_df["save_path"] = save_path
                metrics_df = pd.concat([metrics_df, temp_df], axis=0)
        #save metrics in a csv file
        metrics_df.to_csv(os.path.join(save_path_config.save_artifact_path,'models_metrics.csv'))
        return resu_dict
    
    except Exception as e:
        CustomException(e,sys)

