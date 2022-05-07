import os
import pywt
import glob
import numpy as np

data_dir_ptrain = 'S:\\ds440\\npyresults_positive\\train'

f_path_ptrain = glob.glob(os.path.join(data_dir_ptrain, '*.npy'))

ptrain_records = [np.load(v) for v in f_path_ptrain]
ltrain = len(ptrain_records)

for i in range(0,ltrain):
	ptrain_records[i] = ptrain_records[i].reshape(1,884736)

p_train = ptrain_records[0]

for i in range(1,ltrain):
	p_train = np.concatenate((p_train, ptrain_records[i]),axis = 0)

print(p_train.shape)

np.save('S:\\ds440\\trainingrecords\\positive_train.npy', p_train)




data_dir_ptest = 'S:\\ds440\\npyresults_positive\\test'

f_path_ptest = glob.glob(os.path.join(data_dir_ptest, '*.npy'))

ptest_records = [np.load(v) for v in f_path_ptest]
ltest = len(ptest_records)

for i in range(0,ltest):
	ptest_records[i] = ptest_records[i].reshape(1,884736)

p_test = ptest_records[0]

for i in range(1,ltest):
	p_test = np.concatenate((p_test, ptest_records[i]),axis = 0)

print(p_test.shape)

np.save('S:\\ds440\\trainingrecords\\positive_test.npy', p_test)




data_dir_ntrain = 'S:\\ds440\\npyresults_negative\\train'

f_path_ntrain = glob.glob(os.path.join(data_dir_ntrain, '*.npy'))

ntrain_records = [np.load(v) for v in f_path_ntrain]
ltrain = len(ntrain_records)

for i in range(0,ltrain):
	ntrain_records[i] = ntrain_records[i].reshape(1,884736)

n_train = ntrain_records[0]

for i in range(1,ltrain):
	n_train = np.concatenate((n_train, ntrain_records[i]),axis = 0)

print(n_train.shape)

np.save('S:\\ds440\\trainingrecords\\negative_train.npy', n_train)




data_dir_ntest = 'S:\\ds440\\npyresults_negative\\test'

f_path_ntest = glob.glob(os.path.join(data_dir_ntest, '*.npy'))

ntest_records = [np.load(v) for v in f_path_ntest]
ltest = len(ntest_records)

for i in range(0,ltest):
	ntest_records[i] = ntest_records[i].reshape(1,884736)

n_test = ntest_records[0]

for i in range(1,ltest):
	n_test = np.concatenate((n_test, ntest_records[i]),axis = 0)

print(n_test.shape)

np.save('S:\\ds440\\trainingrecords\\negative_test.npy', n_test)