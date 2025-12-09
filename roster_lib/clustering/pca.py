import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

def find_n_PC(pca_evr,threshold):
    cs = np.cumsum(pca_evr)
    above = cs >= threshold
    n = above.argmax() +1
    print(f"Threshold {threshold:.2f} reached with {n} Principal Components")
    plt.plot(cs);
    plt.plot([0,n-1],[cs[n-1],cs[n-1]], ls = '-', c = 'red');
    plt.plot([n-1,n-1],[cs[0],cs[n-1]], ls = '-', c = 'red');
    plt.xticks(ticks= range(len(cs)),labels=[i+1 for i in range(len(cs))])
    
    return n