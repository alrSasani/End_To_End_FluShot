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
    """An abstract class that defines my raw data configuration that will be 
    used as data source for data ingestion

    """
    @abstractmethod
    def __init__():
        pass

    @abstractmethod
    def get_configuration() -> DataIngestionConfiguration:
        """
        
        Returns:
            DataIngestionConfiguration: this configuration includes addresses of the different files 
            that will be used in ingestion part.
        """
        pass


class GetDataConfigurationFromLocalPath(GetRawDataConfiguration):
    """
    This class returns ingetsted raw data configuration for data located in local directory
    
    Args:
        GetRawDataConfiguration (_type_): _description_

    methods:
        get_configuration: returns the raw data cofiguration
        set_path: sets the path from a local path
    """ 
    def get_configuration(self) -> RawDataFilesConfiguration:
        """a methos that returns ingested data configuration after ingesting data from local path

        Returns:
            RawDataFilesConfiguration: _description_
        """
        self.data_configuration.training_set_features = os.path.join(self.path,"training_set_features.csv")
        self.data_configuration.training_set_labels:str = os.path.join(self.path,"training_set_labels.csv")
        self.data_configuration.test_set_features:str = os.path.join(self.path,"test_set_features.csv")
        self.data_configuration.submission_format:str = os.path.join(self.path,"submission_format.csv")
        return self.data_configuration
    
    def set_path(self,path):
        """sets the path where the raw data are located

        Args:
            path : str : Local path
        """
        self.path = path


class GetDataConfigurationFromAPI(GetRawDataConfiguration):
    """Gets data from API and it a subclass of GetRawDataConfiguration where we need to define 
    get_configurayion method for getting raw data configuration.

    Args:
        
    """
    def __init__(self):
        pass

    def get_configuration(self):
        pass


class DataIngestion:
    """ this class get RawDataFilesConfiguration where we have raw data file paths and 
    use these information to ingest the data
    """
    def __init__(self,raw_data_configuration:RawDataFilesConfiguration):
        """

        Args:
            raw_data_configuration (RawDataFilesConfiguration): get the paths for raw data.
        """
        self.data_ingestion_configuration=DataIngestionConfiguration()
        self.raw_data_configuration = raw_data_configuration

    def ingest_data(self) -> DataIngestionConfiguration:
        """this method ingests the data from RawDataFilesConfiguration

        Returns:
            DataIngestionConfiguration: the paths for different files ingested
        """
        logging.info("ingestion started...")
        try:
            # print('>>> ',self.data_ingestion_configuration.training_set_features)
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
    """This in an interface for getting ingested data configuration from raw data located in local path

    Args:
        DataIngestion (_type_): _description_
    """
    
    def __init__(self,path:str) -> None:
        config_getter = GetDataConfigurationFromLocalPath()
        config_getter.set_path(path)
        self.path = path
        self.raw_data_configuration = config_getter.get_configuration()
        super().__init__(self.raw_data_configuration)

    def get_ingest_configuration(self) -> DataIngestionConfiguration:
        """ingests the data from local path

        Returns:
            DataIngestionConfiguration: configuration for the ingested data.
        """
        try:
            logging.info(f"GettDataIngestFromLocal: getting data from local point {self.path}")
            self.data_ingestion_configuration = self.ingest_data()
            return self.data_ingestion_configuration
        except Exception as e:
            CustomException(e,sys)


def get_data_dataframe(method:DataIngestion):
    pass
