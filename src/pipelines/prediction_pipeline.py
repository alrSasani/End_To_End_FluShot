
import os
from src.training_config import TrainConfiguration
import pandas as pd
from src.utils import get_best_predictor_pipelines
from src.logger import logging
from src.exceptions import CustomException
import sys

def Predict_from_saved_models(df,artifacts_path:TrainConfiguration):
    """
    Function to predict results for a dataframe usinf the best model among the saved models
    """
    logging.info("predicting the results:")
    try:
        csv_path = os.path.join(artifacts_path.save_artifact_path,'models_metrics.csv')
        pipline_h1n1,pipline_seas = get_best_predictor_pipelines(csv_path)

        prediction_1 = pipline_h1n1.predict(df)
        prediction_2 = pipline_seas.predict(df)
        
        df_predictions = pd.DataFrame({
        'H1N1': prediction_1,
        'Seasonal': prediction_2
        })
        return df_predictions
    
    except Exception as e:
        CustomException(e,sys)

def predict_from_mlflow():
    pass


