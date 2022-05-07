from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

X_train = np.load('S:\\ds440\\trainingrecords\\final\\train.npy')
y_train = np.load('S:\\ds440\\trainingrecords\\final\\train_label.npy')
X_test = np.load('S:\\ds440\\trainingrecords\\final\\test.npy')
y_test = np.load('S:\\ds440\\trainingrecords\\final\\test_label.npy')

model = LogisticRegression(max_iter = 50)
model.fit(X_train, y_train)

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)
print(accuracy)
ConfusionMatrix = confusion_matrix(y_test, prediction)
print(ConfusionMatrix)
print(classification_report(y_test, prediction))