import numpy as np
import cv2

a = np.load(r'D:\try\try\szmCode\paconv_\duibikeshihua_hamlyn_85\post_train_sample_num_0_best.npz')
b = np.load(r'D:\try\try\szmCode\paconv_\duibikeshihua_hamlyn_85noise\post_train_sample_num_0_best.npz')

a_points = a['points1']
b_points = b['points1']


# a = np.array([0,0,0])
print(a)