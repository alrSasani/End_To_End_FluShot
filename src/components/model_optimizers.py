
import sys
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from abc import ABC, abstractmethod
from src.logger import logging
from src.exceptions import CustomException
from sklearn.ensemble import RandomForestClassifier

class BaseModelOptimizer(ABC):
    """_summary_. an abstract class that defines the blue print for creating different methods for optimizing 
    different models
    """
    @abstractmethod
    def __init__(self):
        pass

    def optimize(self):
        """this method optimizes the models
        """
        pass


class RandomizeSearchModelOptimizer(BaseModelOptimizer):
    """_summary_ : a calss that uses randomizesearch of sklearn to optimize models.
    """
    def __init__(self, model, param_distributions, cv_nfold=3, iter=10, scoring='Accuracy', random_state=42):
        """_summary_: initialization of the optimizer

        Args:
            model (machine learning model): _description_. machine learning models that implements fit metohds or pipelines
            with preprocessor and machine learning models.
            param_distributions (dictionary): _description_. a dictionary that defines different parameter 
            and their grids (parameter space) in which the model will be oprimized
            cv_nfold (int, optional): _description_.Defaults to 3. number of folds of cross validatin 
            iter (int, optional): _description_. Defaults to 10. number of iteration in randomize search
            scoring (str, optional): _description_. Defaults to 'Accuracy'. Scoring method according to
            how they are defined in sklearn
            random_state (int, optional): _description_. Defaults to 42.
        """
        self.random_state=random_state
        self.scoring = scoring
        self.iter = iter
        self.cv = ShuffleSplit(n_splits=cv_nfold, test_size=0.2, random_state=random_state)
        self.model = model
        self.param_distributions = param_distributions

    def optimize(self, X, y):
        """Optimize the model in paramter space

        Args:
            X (array): features of the data to optimize the model in it
            y (array): labels for the features
        """
        try:
            logging.info(f"Optimizin model {type(self.model).__name__} using RandomizeSearchModelOptimizer")

            grid_search = RandomizedSearchCV(self.model, param_distributions=self.param_distributions, n_iter=20, random_state=self.random_state,
                                        cv=self.cv, scoring=self.scoring)
            grid_search.fit(X, y)  
            self.grid_search = grid_search
            self.best_model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
        except Exception as e:
            CustomException(e,sys)


class GridSearchModelOptimizer(BaseModelOptimizer):
    """_summary_ : a calss that uses gridsearch of sklearn to optimize models.
    """    
    def __init__(self, model, params_grid, cv_nfold=3,  scoring='Accuracy', random_state=42):
        """_summary_: initialization of the optimizer

        Args:
            model (machine learning model): _description_. machine learning models that implements fit metohds or pipelines
            with preprocessor and machine learning models.
            params_grid (dictionary): _description_. a dictionary that defines different parameter 
            and their grids (parameter space) in which the model will be oprimized
            cv_nfold (int, optional): _description_.Defaults to 3. number of folds of cross validatin 
            scoring (str, optional): _description_. Defaults to 'Accuracy'. Scoring method according to
            how they are defined in sklearn
            random_state (int, optional): _description_. Defaults to 42.
        """
        self.random_state=random_state
        self.scoring = scoring
        self.cv = ShuffleSplit(n_splits=cv_nfold, test_size=0.2, random_state=random_state)
        self.model = model
        self.params_grid = params_grid

    def optimize(self, X, y):
        """Optimize the model in paramter space

        Args:
            X (array): features of the data to optimize the model in it
            y (array): labels for the features
        """
        try:
            logging.info(f"Optimizin model {type(self.model).__name__} using GridSearchModelOptimizer")
            grid_search = GridSearchCV(self.model, param_grid=self.params_grid, random_state=self.random_state,
                                        cv=self.cv, scoring=self.scoring)
            grid_search.fit(X, y) 
            self.grid_search =  grid_search
            self.best_model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
        except Exception as e:
            CustomException(e,sys)


class OptunaOptimizer(BaseModelOptimizer):
    def __init__(self):
        pass
    def optimize(self):
        pass