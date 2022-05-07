import os
import pywt
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

data_dir = 'S:\\ds440\\Group\\Bahati1_R'
#remember to change this, or put the pictures into the folder

titles = ['Original', 'Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']

img_files = [v for v in os.listdir(data_dir) if v.startswith('g')]
n_rows = len(img_files)
n_cols = 5

fig = plt.figure(figsize=(15, 3 * n_rows))

for image_idx, image_filename in enumerate(img_files):
    print('image_filename: {}'.format(image_filename))
    img = Image.open(os.path.join(data_dir, image_filename))
    img = np.asarray(img)
    coeffs2 = pywt.dwt2(img, 'bior1.3')

    LL, (LH, HL, HH) = coeffs2

    for i, a in enumerate([img, LL, LH, HL, HH]):
        ax = fig.add_subplot(n_rows, 5, (image_idx * n_cols) + i + 1)
        if i == 0:
            ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
        else:
            ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)   # , cmap=plt.cm.gray

        ax.set_title(image_filename, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

fig.tight_layout()
#plt.savefig('S:\\ds440\\jpegresults\\positive_Abby_R.jpeg')
#remember to change this too

count = 0 #first  time:0, second time: the last number of your prior result
p = len(img_files)
print(p)
for i in range(0,p):
    for j in range(i+1,p):
        result = np.concatenate((img[i],img[j]),axis=0)
        print(len(result))
        print(len(result[0]))
        count += 1
        #np.save('S:\\ds440\\npyresults_positive\\positive_ex_'+str(count)+".npy",result)
       # print("%.f" % i,"%.f" % j,result)0
