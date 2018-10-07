#Stephen Pretto (spretto2), HW6, 10/2018

import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score

#Create the pipe
#CV/Grid Search
iris = load_iris()
X = iris.data
y = iris.target

tree = DecisionTreeClassifier(max_depth=6)

test_scores = []
train_scores = []

for i in range(1,11):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=i)
    tree.fit(X_train, y_train)
    train_scores.append(tree.score(X_train, y_train))
    test_scores.append(tree.score(X_test, y_test))
    
print(train_scores)
print(test_scores)
print('Test (out-of-sample) accuracy: %.3f +/- %.3f' % (np.mean(test_scores), np.std(test_scores)))


print("Cross_Val_Score")
cross_val_tree = cross_val_score(tree, X_train,y_train, cv=10, scoring='accuracy')
print('Accuracy: %.3f +/- %.3f' % (np.mean(cross_val_tree), np.std(cross_val_tree)))

tree.score(X_test, y_test)

param_grid = {'max_depth': np.arange(1,10)} #Parameter grid
gs = GridSearchCV(tree, param_grid, scoring='accuracy', cv=10, n_jobs=-1)
gs.fit(X, y)
print(gs.best_score_)
print(gs.best_params_)


print("My name is Stephen Pretto")
print("My NetID is: spretto2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")