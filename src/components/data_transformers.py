from dataclasses import dataclass
from src.logger import logging
import pandas as pd
import sys
from src.exceptions import CustomException
from src.components.data_ingestion import DataIngestionConfiguration
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler, OneHotEncoder ,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from typing import List
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
# from category_encoders import OrdinalEncoder
# OrdinalEncoder(mapping=ordinal_mapping)
from src.training_config import SchemaDataType, DataTransformationConfiguration


class GetPipeline(ABC):
    """An abstract class tha defines my general preprocessor blue prints

    """
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_transformer(self):
        """returns a transformer
        """
        pass


class GetDefaultsPreprocessor(GetPipeline):
    """A default preprocessor that will be useed in machine learning models
    """
    def __init__(self,schemat_dtypes=SchemaDataType()):
        """

        Args:
            schemat_dtypes : _description_. Defaults to SchemaDataType() where different data are
            defined types for different columns.
        """
        self.schemat_dtypes = schemat_dtypes

    def get_transformer(self):
        """a method to get transformer

        Returns:
            a preprocessor with fit and fit_transform methods
        """
        logging.info(""" GetDefaultsPreprocessor: Getting default preprocessor with transformer that uses most frequent
                        value to fill Null values and uses Ordinal Encoder for categorical values
                        and also StandardScaler for all features""")
        try:
            cat_columns = self.schemat_dtypes.cat_columns+self.schemat_dtypes.ordinal_obj_columns
            num_columns = self.schemat_dtypes.boolean_columns+self.schemat_dtypes.ordinal_num_columns

            nume_pipline = make_pipeline(
                SimpleImputer(strategy='most_frequent'),
                    #    StandardScaler()
                    
            )
            cat_pipline = make_pipeline(
                
                    SimpleImputer(strategy="most_frequent"),
                    OrdinalEncoder(),
                    # StandardScaler()
                
            )
            
            preprocessor = ColumnTransformer(
                transformers=[('cat_transformet',cat_pipline,cat_columns),
                            ('numerical_transformet',nume_pipline,num_columns)],
                remainder='passthrough'
            )
            self.preprocessor = preprocessor

            return self.preprocessor
        
        except Exception as e:
            CustomException(e,sys)


class SimpleNullFiller(BaseEstimator, TransformerMixin):
    """A transformer to fill null values of categorical types with None and numerical columns with -1
    methods:
        fit:
        transform: transforms null values according to dictionary provided in __init__ method
    """
    def __init__(self,key_dic={'O':'None','num':-1}):
        self.key_dic = key_dic

    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        self.X = X
        for col_name in X.columns:
            if X[col_name].dtype =='O':
                X[col_name]= X[col_name].fillna(self.key_dic['O'])
            else:
                X[col_name] = X[col_name].fillna(self.key_dic['num'])

        self.input_features = X.columns
        return X
    
    def get_feature_names_out(self,X=None,y=None):
        # if X:
        self.input_eatures = self.X.columns
        
        return self.input_features
                
  
class GetSimpleTransformer(GetPipeline):
    """A transformer to fill null values with Simple null filler and transform categorical columns 
    using ordinal encoder

    """
    def __init__(self,schemat_dtypes=SchemaDataType()):
        self.schemat_dtypes = schemat_dtypes   
    def get_transformer(self):
        logging.info(""" GetSimpleTransformer ... """)
        try:
            cat_columns = self.schemat_dtypes.cat_columns+self.schemat_dtypes.ordinal_obj_columns
            num_columns = self.schemat_dtypes.boolean_columns+self.schemat_dtypes.ordinal_num_columns

            simple_null_filler_pip = SimpleNullFiller()

            nume_pipline = make_pipeline(
                StandardScaler()         
            )

            cat_pipline = make_pipeline(
                    OrdinalEncoder(),
                    StandardScaler()   
            )
            
            temp_preprocessor = ColumnTransformer(
                transformers=[('cat_transformet',cat_pipline,cat_columns),
                            ('numerical_transformet',nume_pipline,num_columns)],
                remainder='passthrough'
            )

            preprocessor = Pipeline(steps=[
                ('SimNullfiller',simple_null_filler_pip),
                ('Columns_trans',temp_preprocessor)
            ])

            self.preprocessor = preprocessor

            return self.preprocessor 
        
        except Exception as e:
            CustomException(e,sys)


class GetCatboostTransformer(GetPipeline):
    """a class that creates transformer for catboost classifiyer where we only fill null values
    using SimpleNullFiller and scale numerical columns.
    """
    def __init__(self,schemat_dtypes=SchemaDataType()):
        self.schemat_dtypes = schemat_dtypes
    
    def get_transformer(self):
        logging.info("GetCatboostTransformer")
        try:
            num_columns = self.schemat_dtypes.boolean_columns+self.schemat_dtypes.ordinal_num_columns

            simple_null_filler_pipe = SimpleNullFiller()

            
            temp_preprocessor = ColumnTransformer(
                transformers=[
                            ('numerical_transformet',StandardScaler(),num_columns)],
                remainder='passthrough'
            )

            preprocessor = Pipeline(steps=[
                ('SimNullfiller',simple_null_filler_pipe),
                ('Columns_trans',temp_preprocessor)

            ])

            self.preprocessor = preprocessor

            return self.preprocessor
        except Exception as e:
            CustomException(e,sys)


class SimpleOrdinalMapper(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        pass
    
    def get_feature_names_out(self,X=None,y=None):
        pass

