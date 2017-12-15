

#methods for scaling data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)  # Don't cheat - fit only on training data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)  # apply same transformation to test data

# Fit to data and predict using pipelined scaling, GNB and PCA.
from sklearn.pipeline import make_pipeline
std_clf = make_pipeline(StandardScaler(), PCA(n_components=2), GaussianNB())
std_clf.fit(X_train, y_train)
pred_test_std = std_clf.predict(X_test)


import numpy as np
import matplotlib.pyplot as plt

N=100
p=10
N_train=N/5
from sklearn import datasets
X,y = datasets.make_classification(n_samples=N,n_features=p,
	n_informative=p/2)
X_train=X[:N_train]
X_test=X[N_train:]
y_train=y[:N_train]
y_test=y[N_train:]

#train some classifiers
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)


