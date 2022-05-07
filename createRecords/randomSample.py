import os
import pywt
import glob
import numpy as np
import random

data_dir = 'S:\\ds440\\npyresults_negative'
f_path = glob.glob(os.path.join(data_dir, '*.npy'))
records = [np.load(v) for v in f_path]

new = random.sample(records, 864)

for i in range(0,864):
	np.save('S:\\ds440\\negativeresult\\negative_ex_'+str(i+1)+".npy",new[i])