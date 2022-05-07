from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import numpy as np

X_train = np.load('S:\\ds440\\trainingrecords\\final\\train.npy')
y_train = np.load('S:\\ds440\\trainingrecords\\final\\train_label.npy')
X_test = np.load('S:\\ds440\\trainingrecords\\final\\test.npy')
y_test = np.load('S:\\ds440\\trainingrecords\\final\\test_label.npy')

clf = RandomForestClassifier(n_estimators = 160, max_depth = 80, random_state = 10)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
ConfusionMatrix = confusion_matrix(y_test, y_pred)
print(ConfusionMatrix)
print(classification_report(y_test, y_pred))

#param_grid = {"n_estimators": range(80,200,10), "max_depth": range(50,100,10)}
#gridsearch = GridSearchCV(estimator = RandomForestClassifier(random_state = 10), param_grid, cv = 5).fit(X_train, y_train)
#gridsearch.best_params_