import pandas as pd
import numpy as np
import warnings

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import root_mean_squared_error
from scipy.stats import uniform, loguniform, randint

# Classification models :
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
# Baseline 
from sklearn.dummy import DummyRegressor


def initiate_model_list(X_train):

    models_names = ["Baseline",
                    "Linear Regression",
                    "Ridge",
                    "Lasso",
                    "ElasticNet",
                    "kNN Regressor",                
                    "Random Forest",
                    "Ada Boost",
                    "XGB Regressor",
                    'SVR - Linear',
                    'SVR - Polynomial',
                    'SVR - RBF',
                    "PLS - DA",                                     
                    
    ]

    models = [
                DummyRegressor(),
                LinearRegression(),
                Ridge(),
                Lasso(),
                ElasticNet(),
                KNeighborsRegressor(),
                RandomForestRegressor(),
                AdaBoostRegressor(),
                XGBRegressor(),
                SVR(kernel = "linear"),
                SVR(kernel = "poly"),
                SVR(kernel = "rbf"),
                PLSRegression(),  
    ]
    
    hp = [{},
            {},
            {'alpha':loguniform(0.01,100)},
            {'alpha':loguniform(0.01,100)},
            {'alpha':loguniform(0.01,100), 'l1_ratio':uniform(0,1)},
            {'n_neighbors':randint(2,60)},
            {'n_estimators':randint(10,500), 'criterion':['squared_error', 'absolute_error', 'friedman_mse', 'poisson'], 'max_depth': randint(1,20), 'min_samples_leaf' : randint(1,40)},
            {'n_estimators':randint(10,500), 'learning_rate': loguniform(1e-3,10)},
            {'eta':uniform(0,1), 'gamma':loguniform(1e-4,1000), 'max_depth': randint(1,20), 'lambda':loguniform(1e-2,10)},
            {'kernel':['linear'], 'C':loguniform(1,100)},
            {'kernel':['poly'], 'C':loguniform(1,100), 'degree':randint(2,3), 'gamma':loguniform(1e-4, 1)},
            {'kernel':['rbf'], 'C':loguniform(1,100), 'gamma':loguniform(1e-4, 1)},
            {'n_components':randint(1,X_train.shape[1])},  
    ]
    
    return models_names, models, hp

def compare_models_w_hpo(X_train, 
                        y_train, 
                        X_test, 
                        y_test, 
                        n_iter = 100, 
                        verbose = True):

    results = []
    asc = True
    kpi = 'RMSE'
    models_names, models, hp = initiate_model_list(X_train)
    for model_name, model,hp in zip(models_names, models,hp):
        if verbose :
            print(f"Testing {model_name}", end='\r')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            hrs = RandomizedSearchCV(model, hp, cv = 5, scoring= ['neg_mean_squared_error'], 
                                     n_jobs=-1, n_iter = n_iter,refit='neg_mean_squared_error').fit(X_train, y_train)
        _y_pred = hrs.predict(X_test)
        result = {
            'Model': model_name,
            'Best_hyperparameters' : hrs.best_params_,
            'RMSE' : root_mean_squared_error(y_test,_y_pred), 
        }
        if verbose :
            print(f"Tested {model_name:<20} - reached {kpi} of {result[kpi]:.2f} over the test set", end = '\n')
        results.append(result) 

    comparing_models = pd.DataFrame(results)

    return comparing_models.set_index('Model').sort_values(by = kpi,ascending = asc)
