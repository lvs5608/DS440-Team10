import os
import numpy as np

data_dir_train = 'S:\\ds440\\trainingrecords'

ptrain = np.load('S:\\ds440\\trainingrecords\\positive_train.npy')
ntrain = np.load('S:\\ds440\\trainingrecords\\negative_train.npy')

train = np.concatenate((ptrain, ntrain),axis = 0)
print(train.shape)
np.save('S:\\ds440\\trainingrecords\\final\\train.npy', train)


ptest = np.load('S:\\ds440\\trainingrecords\\positive_test.npy')
ntest = np.load('S:\\ds440\\trainingrecords\\negative_test.npy')

test = np.concatenate((ptest, ntest),axis = 0)
print(test.shape)
np.save('S:\\ds440\\trainingrecords\\final\\test.npy', test)


plabel = np.load('S:\\ds440\\trainingrecords\\positive_label.npy')
nlabel = np.load('S:\\ds440\\trainingrecords\\negative_label.npy')

train_label = np.concatenate((plabel, nlabel),axis = None)
print(train_label.shape)
np.save('S:\\ds440\\trainingrecords\\final\\train_label.npy', train_label)


ptlabel = np.load('S:\\ds440\\trainingrecords\\ptest_label.npy')
ntlabel = np.load('S:\\ds440\\trainingrecords\\ntest_label.npy')

test_label = np.concatenate((ptlabel, ntlabel),axis = None)
print(test_label.shape)
np.save('S:\\ds440\\trainingrecords\\final\\test_label.npy', test_label)