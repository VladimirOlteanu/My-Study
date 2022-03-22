#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import print_function

import argparse
import os
import pandas as pd

from sklearn import tree
from sklearn.externals import joblib


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here. In this simple example we are just including one hyperparameter.
    parser.add_argument('--max_depth', type=int, default=3)
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--max_features', type=int, default=None)

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    file = os.path.join(args.train, "training_data.csv")
    dataset = pd.read_csv(file, engine="python")

     # labels are in the first column
    y_train =dataset.iloc[:, 0]
    x_train =dataset.iloc[:, 1:]

    # Hyperparameters
    
    max_depth = args.max_depth
    n_estimators = args.n_estimators
    learning_rate = args.learning_rate
    max_features = args.max_features

    # Now use scikit-learn's decision tree classifier to train the model.
    clf = GradientBoostingClassifier.DecisionTreeClassifier(max_depth=max_depth,n_estimators=n_estimators,learning_rate=learning_rate,max_features=max_features)
    clf = clf.fit(x_train, y_train)

    # Print the coefficients of the trained classifier, and save the coefficients
    joblib.dump(clf, os.path.join(args.model_dir, "model.joblib"))


def model_fn(model_dir):
    """Deserialized and return fitted model

    Note that this should have the same name as the serialized model in the main method
    """
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf

