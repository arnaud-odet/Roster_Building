import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

def find_best_decision_boundary(y_true, y_prob, step:float = 0.05):
    dbs = np.arange(0,1+step,step)
    f1s = np.array([f1_score(y_true= y_true, y_pred = (y_prob > db).astype(int)) for db in dbs])
    return dbs[np.argmax(f1s)], max(f1s)