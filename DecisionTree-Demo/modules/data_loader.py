import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

def load_sample_data():
    # Load the Iris dataset
    iris = load_iris()
    data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    data['target'] = iris.target
    return data

def load_synthetic_data():
    # Generate a synthetic dataset
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=200, n_features=4, n_informative=3, 
        n_redundant=0, random_state=42, class_sep=2
    )
    data = pd.DataFrame(X, columns=[f"Feature {i+1}" for i in range(X.shape[1])])
    data['target'] = y
    return data
