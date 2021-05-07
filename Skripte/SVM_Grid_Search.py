from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

path = r"D:\Dataframes\20PCs\bl693_no_white_Pop05.csv"
dataframe = pd.read_csv(path)

X = []
y = []

for index, row in dataframe.iterrows():
    X.append(row[2:-1].tolist())
    # Binary Labels for activity
    if row[-1] > 0:
        y.append(1)
    else:
        y.append(0)

dataframe = None
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Grid Search
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV
# https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html#sphx-glr-auto-examples-svm-plot-rbf-parameters-py
# https://machinelearninggeek.com/grid-search-in-scikit-learn/

c = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1, 10, 25, 50, 100, 1000, 10000]
parameters = {'kernel':['rbf'], 'C':c, 'gamma': c}
clf = GridSearchCV(SVC(), parameters, scoring = 'f1', cv = 5)
clf.fit(X_train, y_train)

print(clf.best_estimator_)
print(clf.score(X_test, y_test))

"""
F1 Score
SVC(C=1000, gamma=100)
0.2916666666666667

Accuracy
SVC(C=10, gamma=25)
0.7869158878504673
"""
