
import sys
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from abc import ABC, abstractmethod
from src.logger import logging
from src.exceptions import CustomException
from sklearn.ensemble import RandomForestClassifier

class BaseModelOptimizer(ABC):
    @abstractmethod
    def __init__(self):
        pass

    def optimize(self):
        pass


class RandomizeSearchModelOptimizer(BaseModelOptimizer):
    def __init__(self, model, param_distributions, cv_nfold=3, iter=10, scoring='Accuracy', random_state=42):
        self.random_state=random_state
        self.scoring = scoring
        self.iter = iter
        self.cv = ShuffleSplit(n_splits=cv_nfold, test_size=0.2, random_state=random_state)
        self.model = model
        self.param_distributions = param_distributions

    def optimize(self, X, y):
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
    def __init__(self, model, params_grid, cv_nfold=3, iter=10, scoring='Accuracy', random_state=42):
        self.random_state=random_state
        self.scoring = scoring
        self.iter = iter
        self.cv = ShuffleSplit(n_splits=cv_nfold, test_size=0.2, random_state=random_state)
        self.model = model
        self.params_grid = params_grid

    def optimize(self, X, y):
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