
import os
from src.training_config import TrainConfiguration
import pandas as pd
from src.utils import load_object
from sklearn.pipeline import make_pipeline
from src.logger import logging
from src.exceptions import CustomException

def Predict_from_saved_models(df,artifacts_path:TrainConfiguration):
    """
    Function to predict results for a dataframe usinf the best model among the saved models
    """
    logging.info("predicting the results:")
    try:
        csv_path = os.path.join(artifacts_path.save_artifact_path,'models_metrics.csv')
        #TODO choose the best model
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
        prediction_1 = pipline_1.predict(df)
        prediction_2 = pipline_2.predict(df)
        
        df_predictions = pd.DataFrame({
        'H1N1': prediction_1,
        'Seasonal': prediction_2
        })
        return df_predictions
    except Exception as e:
        Cus

def predict_from_mlflow():
    pass

def DataFormat():
    pass
