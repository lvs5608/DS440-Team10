import numpy as np


positive_label = np.full((648,), 1)
ptest_label = np.full((216,), 1)
negative_label = np.full((648,), 0)
ntest_label = np.full((216,), 0)

print(positive_label)
print(negative_label)
print(positive_label.shape)
print(negative_label.shape)
print(ptest_label.shape)
print(ntest_label.shape)

np.save('S:\\ds440\\trainingrecords\\positive_label.npy', positive_label)
np.save('S:\\ds440\\trainingrecords\\negative_label.npy', negative_label)
np.save('S:\\ds440\\trainingrecords\\ptest_label.npy', ptest_label)
np.save('S:\\ds440\\trainingrecords\\ntest_label.npy', ntest_label)