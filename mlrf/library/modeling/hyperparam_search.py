import sys
import os

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

class HyperparamsSearch:
    def __init__(self, hyperparams, model, verbose=0, cv=None, n_jobs=None):
        self.hyperparams = hyperparams
        self.model = model
        self.verbose = verbose
        self.cv = cv
        self.n_jobs = n_jobs
        
    def set_verbose_level(self, verbose):
        self.verbose = verbose
        
    def set_cv_fold(self, cv):
        self.cv = cv
        
    def set_njobs(self, n_jobs):
        self.n_jobs = n_jobs
        
    def grid_search(self, X_train, y_train):
        gs = GridSearchCV(estimator=self.model, param_grid=self.hyperparams,
                                   cv=self.cv, n_jobs=self.n_jobs, verbose=self.verbose)
        gs.fit(X_train, y_train)
        
        return gs.best_estimator_, gs.best_params_
    
    def random_search(self):
        rs = RandomizedSearchCV(estimator=self.model, param_grid=self.hyperparams,
                                   cv=self.cv, n_jobs=self.n_jobs, verbose=self.verbose)
        rs.fit(X_train, y_train)
        
        return rs.best_estimator_, rs.best_params_
        