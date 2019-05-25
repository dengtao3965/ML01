import numpy as np
from sklearn import svm,datasets
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


iris=datasets.load_iris()
X,y=iris.data,iris.target
clf=svm.SVC(kernel='linear',C=1,random_state=0)

n_folds=5
kf=KFold(n_folds,shuffle=True,random_state=42).get_n_splits(X)
scores=cross_val_score(clf,X,y,scoring='precision_macro',cv=kf)

print(scores)