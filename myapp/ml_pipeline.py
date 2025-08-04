import numpy as np
import pickle
import os

# Load models (only once)
rmodel_path = os.path.join('myapp', 'models', 'ridge_regression_model.pkl')
dtmodel_path = os.path.join('myapp', 'models', 'decision_tree_model.pkl')

with open(rmodel_path, 'rb') as f:
    rmodel = pickle.load(f)

with open(dtmodel_path, 'rb') as f:
    dtmodel = pickle.load(f)

def predict_fare(features):
    """
    features: list of numeric values, matching model input order
    """
    X = np.array([features])
    rprediction = rmodel.predict(X)
    dtprediction = dtmodel.predict(X)
    return rprediction[0], dtprediction[0]
