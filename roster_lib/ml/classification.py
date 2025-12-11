import pandas as pd
import numpy as np
import warnings

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, cohen_kappa_score
from scipy.stats import uniform, loguniform, randint

# Classification models :
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
# Baseline 
from sklearn.dummy import DummyClassifier


def initiate_class_model_list(X_train,n_class:int):

    clas_models_names = ["Logistic Regression",
                    "k - Nearest Neighbors Classifier",                
                    "Random Forest",
                    "Ada Boost",
                    "XGB Classifier",
                    'Support Vector Classifier - Linear kernel',
                    'Support Vector Classifier - Polynomial kernel',
                    'Support Vector Classifier - RBF kernel',
                    "Partial Least Squares - Discriminant Analysis",                                     
                    "Baseline most_frequent",
    ]

    clas_models = [LogisticRegression(max_iter=1000),
                KNeighborsClassifier(),
                RandomForestClassifier(),
                AdaBoostClassifier(),
                XGBClassifier(),
                SVC(kernel = "linear"),
                SVC(kernel = "poly"),
                SVC(kernel = "rbf"),
                PLSRegression(),  
                DummyClassifier(),
    ]
    
    clas_hp = [{'penalty':['l1','l2','elasticnet',None], 'C':loguniform(1,1000), 'solver':['saga']},
                {'n_neighbors':randint(2,60)},
                {'n_estimators':randint(10,500), 'criterion':['gini', 'entropy', 'log_loss'], 'max_depth': randint(1,20), 'min_samples_leaf' : randint(1,40)},
                {'n_estimators':randint(10,500), 'learning_rate': loguniform(1e-3,10)},
                {'eta':uniform(0,1), 'gamma':loguniform(1e-4,1000), 'max_depth': randint(1,20), 'lambda':loguniform(1e-2,10)},
                {'kernel':['linear'], 'C':loguniform(1,100)},
                {'kernel':['poly'], 'C':loguniform(1,100), 'degree':randint(2,3), 'gamma':loguniform(1e-4, 1)},
                {'kernel':['rbf'], 'C':loguniform(1,100), 'gamma':loguniform(1e-4, 1)},
                {'n_components':randint(1,X_train.shape[1])},  
                {},
    ]
    
    return clas_models_names,clas_models, clas_hp

def compare_models_w_hpo(X_train, 
                        y_train, 
                        X_test, 
                        y_test, 
                        n_iter = 100, 
                        verbose = True):

    results = []
    asc = False
    kpi = 'ROC-AUC'
    
    n_class = len(np.unique(y_train))
    clas_models_names, clas_models, clas_hp = initiate_class_model_list(X_train,n_class)
    for model_name, model,hp in zip(clas_models_names, clas_models,clas_hp):
        if verbose :
            print(f"Testing {model_name}", end='\r')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            hrs = RandomizedSearchCV(model, hp, cv = 5, scoring= ['f1'], 
                                     n_jobs=-1, n_iter = n_iter,refit='f1').fit(X_train, y_train)
        _y_pred_prob = hrs.predict(X_test)
        _y_pred = np.array([ 1 if pred > 0.5 else 0 for pred in _y_pred_prob])      
        result = {
            'Model': model_name,
            'Best_hyperparameters' : hrs.best_params_,
            'Accuracy' : accuracy_score(y_test,_y_pred), 
            'ROC-AUC': roc_auc_score(y_test, _y_pred_prob),
            'F1-score' : f1_score(_y_pred,y_test),
            'Cohen-Kappa' : cohen_kappa_score(y_test, _y_pred) 
        }
        if verbose :
            print(f"Tested {model_name} - reached {kpi} of {np.round(result[kpi],3)} over the test set", end = '\n')
        results.append(result) 

    comparing_models = pd.DataFrame(results)

    return comparing_models.set_index('Model').sort_values(by = kpi,ascending = asc)
