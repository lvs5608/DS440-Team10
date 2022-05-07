from flaml import AutoML
import numpy as np

X_train = np.load('S:\\ds440\\trainingrecords\\final\\train.npy')
y_train = np.load('S:\\ds440\\trainingrecords\\final\\train_label.npy')
#X_test = np.load('S:\\ds440\\trainingrecords\\final\\test.npy')
#y_test = np.load('S:\\ds440\\trainingrecords\\final\\test_label.npy')


automl = AutoML()
automl_settings = {"metric": 'accuracy', "task": 'classifiction'}
automl.fit(X_train, y_train, **automl_settings)

print(automl.predict_proba(X_train))
print(automl.model.estimator)