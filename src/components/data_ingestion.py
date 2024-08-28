# data should be loaded and saved to tarin and test to a particular directory
from dataclasses import dataclass
import os
import pandas as pd
from src.logger import logging
from src.exceptions import CustomException
import sys
from abc import ABC, abstractmethod
from src.training_config import DataIngestionConfiguration, RawDataFilesConfiguration 

class GetRawDataConfiguration(ABC):
    '''
    Abstract class to define different methods od getting data to be digested
    '''
    @abstractmethod
    def __init__():
        pass

    @abstractmethod
    def get_configuration():
        pass


class GetDataConfigurationFromLocalPath(GetRawDataConfiguration):
    '''
    a class to get RawDataConfiguration from a path on local machine
    '''
    def __init__(self,path:str):
        self.data_configuration = RawDataFilesConfiguration
        self.path = path

    def get_configuration(self):
        self.data_configuration.training_set_features = os.path.join(self.path,"training_set_features.csv")
        self.data_configuration.training_set_labels:str = os.path.join(self.path,"training_set_labels.csv")
        self.data_configuration.test_set_features:str = os.path.join(self.path,"test_set_features.csv")
        self.data_configuration.submission_format:str = os.path.join(self.path,"submission_format.csv")
        return self.data_configuration


class GetDataConfigurationFromAPI(GetRawDataConfiguration):
    '''
    a class to get RawDataConfiguration from an API
    '''
    def __init__(self):
        pass
    def get_configuration(self):
        pass


class DataIngestion:
    """
    a class to save the data to the to the DataIngestionConfiguration:
    """
    def __init__(self,raw_data_configuration:RawDataFilesConfiguration):
        self.data_ingestion_configuration=DataIngestionConfiguration()
        self.raw_data_configuration = raw_data_configuration

    def ingest_data(self) -> DataIngestionConfiguration:
        logging.info("ingestion started...")
        try:
            print('>>> ',self.data_ingestion_configuration.training_set_features)
            os.makedirs(os.path.dirname(self.data_ingestion_configuration.training_set_features),exist_ok=True)

            logging.info("reading files from raw data directory and saving to artifacts.")

            training_set_features = pd.read_csv(self.raw_data_configuration.training_set_features,index_col='respondent_id')
            training_set_features.to_csv(self.data_ingestion_configuration.training_set_features,header=True,index=True)

            training_set_labels = pd.read_csv(self.raw_data_configuration.training_set_labels,index_col='respondent_id')
            training_set_labels.to_csv(self.data_ingestion_configuration.training_set_labels,header=True,index=True)

            test_set_features = pd.read_csv(self.raw_data_configuration.test_set_features,index_col='respondent_id')
            test_set_features.to_csv(self.data_ingestion_configuration.test_set_features,header=True,index=True)

            submission_format = pd.read_csv(self.raw_data_configuration.submission_format,index_col='respondent_id')
            submission_format.to_csv(self.data_ingestion_configuration.submission_format,header=True,index=True)

            logging.info("Files saved to artifacts directory.")
            return self.data_ingestion_configuration
        
        except Exception as e:
            raise CustomException(e,sys)


class GettDataIngestFromLocal(DataIngestion):
    '''
    an interface to digest the data and save it to a DataIngestionConfiguration from a local
    machine
    '''
    
    def __init__(self,path:str) -> None:
        config_getter = GetDataConfigurationFromLocalPath(path)
        self.path = path
        self.raw_data_configuration = config_getter.get_configuration()
        super().__init__(self.raw_data_configuration)

    def get_ingest_configuration(self) -> DataIngestionConfiguration:
        try:
            logging.info(f"GettDataIngestFromLocal: getting data from local point {self.path}")
            self.data_ingestion_configuration = self.ingest_data()
            return self.data_ingestion_configuration
        except Exception as e:
            CustomException(e,sys)


def get_data_dataframe(method:DataIngestion):
    pass
