from dataclasses import dataclass
import os
from sklearn.linear_model import RidgeClassifier,SGDClassifier
from sklearn.ensemble import (ExtraTreesClassifier ,RandomForestClassifier,
                               AdaBoostClassifier, GradientBoostingClassifier)
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier  
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

"""
Defining parameter to be use in the training mode include grid paramters, 
models and path to data as well as features data types
"""

ABSOLOUT_PATH = "/Users/alirezasasani/Documents/Programming/DS_Projects/DS_Competitions/End_To_End_Flushot"

@dataclass
class RawDataFilesConfiguration():
    """
    a class that contains information of the files we need to digest.
    """
    training_set_features:str 
    training_set_labels:str 
    test_set_features:str
    submission_format:str

@dataclass
class DataIngestionConfiguration():
    """
    a class to hold the saving path of the data after digestion.
    """
    training_set_features:str = os.path.join(ABSOLOUT_PATH,"artifact","ingested_data","training_set_features.csv")
    training_set_labels:str = os.path.join(ABSOLOUT_PATH,"artifact","ingested_data","training_set_labels.csv")
    test_set_features:str = os.path.join(ABSOLOUT_PATH,"artifact","ingested_data","test_set_features.csv")
    submission_format:str = os.path.join(ABSOLOUT_PATH,"artifact","ingested_data","submission_format.csv")
  
@dataclass
class SchemaDataType():
    """
    a class that contains different data type of the features:
    """
    cat_columns = [
        'race', 'sex', 'marital_status', 'employment_status', 'hhs_geo_region',
        'census_msa', 'employment_industry', 'employment_occupation', 'rent_or_own'
    ]

    ordinal_num_columns = [
        'h1n1_concern', 'h1n1_knowledge', 'opinion_h1n1_vacc_effective', 'opinion_h1n1_risk',
        'opinion_h1n1_sick_from_vacc', 'opinion_seas_vacc_effective', 'opinion_seas_risk', 
        'opinion_seas_sick_from_vacc', 'household_adults', 'household_children',
    ]

    ordinal_obj_columns = ['age_group', 'education', 'income_poverty']

    boolean_columns = [
        'behavioral_antiviral_meds', 'behavioral_avoidance', 'behavioral_face_mask', 
        'behavioral_wash_hands', 'behavioral_large_gatherings',  'behavioral_outside_home',
        'behavioral_touch_face', 'doctor_recc_h1n1',  'doctor_recc_seasonal', 'chronic_med_condition', 
        'child_under_6_months', 'health_worker', 'health_insurance',
    ]

    ordinal_mapping = [
        {
            'col': 'age_group',
            'mapping': {
                '18 - 34 Years': 0, 
                '35 - 44 Years': 1, 
                '45 - 54 Years': 2,
                '55 - 64 Years': 3, 
                '65+ Years': 4,
            }
        },
        {
            'col': 'education',
            'mapping': {
                '< 12 Years': 0, 
                '12 Years': 1, 
                'College Graduate': 2, 
                'Some College': 3
            }
        },
        {
            'col': 'income_poverty',
            'mapping': {
                'Below Poverty': 0, 
                '<= $75,000, Above Poverty': 1,
                '> $75,000': 2
            }
        }
    ]

@dataclass
class DataTransformationConfiguration():
    pass

# A dictionary that include all the models that we want to test:
MODELS_TO_BE_TRAINED = {
    'cat_Boost' : CatBoostClassifier(verbose=False),
    # 'ridge' : RidgeClassifier(),
    # 'sgd' : SGDClassifier(),
    # 'extra_tree' : ExtraTreesClassifier(),
    # 'kneighbors' : KNeighborsClassifier(),
    # 'decision_tree' : DecisionTreeClassifier(),
    'random_forest' : RandomForestClassifier(),
    # 'adaboost' : AdaBoostClassifier(algorithm='SAMME'),
    # 'gradient_boosting' : GradientBoostingClassifier() 
    }

# A dictionary that include the parameter grid for wach of the models we want to optimize:
PARAMETER_GRID = {
    'random_forest' : {
    'n_estimators':[100,200,300,400],
    'max_depth':np.arange(1,100,1),
    'min_samples_split':np.arange(2,100,1),
    'min_samples_leaf':np.arange(1,100,1),
    # 'min_weight_fraction_leaf':1,
    # 'max_leaf_nodes':1,
    # 'max_samples':1,
    # 'min_impurity_decrease':1,
    'bootstrap':[True,False],
    # 'class_weight':1,
    # 'oob_score':1,
    # 'random_state':1,
    # 'n_jobs':1,
    # 'max_features' : ["sqrt", "log2",  int , float],
    # 'criterion' : ["gini", "entropy", "log_loss"]
    },
    'cat_Boost' : {
        'iterations': [100,200], #[100,200,300,500,1000,1200,1500],
        'learning_rate':[ 0.001, 0.3],
        #'random_strength': int, #trial.suggest_int("random_strength", 1,10),
        #'bagging_temperature':int, #trial.suggest_int("bagging_temperature", 0,10),
        #'max_bin':int, #trial.suggest_categorical('max_bin', [4,5,6,8,10,20,30]),
        'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide'],
        #'min_data_in_leaf':int, #trial.suggest_int("min_data_in_leaf", 1,10),
        'od_type' : ["Iter"],
        'od_wait' : [100],
        #"depth": int, #trial.suggest_int("max_depth", 2,10),
        # "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1e-8, 100),
        'one_hot_max_size':[5,10,12,100,500,1024],
        'custom_metric' : ['AUC'],
        "loss_function": ["Logloss"],
        'auto_class_weights':['Balanced', 'SqrtBalanced'],
    }
}

# a class that contains atrifacts path.
@dataclass
class TrainConfiguration:
    save_artifact_path:str = os.path.join(ABSOLOUT_PATH,"artifact")

#path of the raw data.
@dataclass
class RawDataPathConfig:
   raw_data_path = os.path.join(ABSOLOUT_PATH,'raw_data')

